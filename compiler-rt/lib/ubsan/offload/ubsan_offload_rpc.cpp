//===-- ubsan_offload_rpc.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handling of the RPC server, the communication channel the GPU uses to send
// sanitizer reports to be serviced on the host.
//
//===----------------------------------------------------------------------===//

#include "ubsan_offload_rpc.h"

#include <dlfcn.h>
#include <pthread.h>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "shared/rpc.h"
#include "ubsan_offload.h"
#include "ubsan_offload_hsa.h"

using namespace __sanitizer;

namespace __sanitizer {

int internal_pthread_create(void *Th, void *Attr, void *(*Callback)(void *),
                            void *Param) {
  return pthread_create(reinterpret_cast<pthread_t *>(Th),
                        reinterpret_cast<const pthread_attr_t *>(Attr),
                        Callback, Param);
}

int internal_pthread_join(void *Th, void **Ret) {
  return pthread_join(reinterpret_cast<pthread_t>(Th), Ret);
}

} // namespace __sanitizer

namespace __ubsan {
namespace {

struct DeviceRpc {
  void *Buffer;
  u32 Ports;
  u32 Lanes;
};

Mutex RpcMutex;
InternalMmapVectorNoCtor<DeviceRpc> Devices;
void *ServerThread;
bool Offload;
atomic_uint8_t Stop;

// Receives reports from the device and services them.
uint32_t handleReport(void *PortPtr, uint32_t) {
  auto &Port = *reinterpret_cast<rpc::Server::Port *>(PortPtr);
  if (Port.get_opcode() != UBSAN_OFFLOAD_REPORT_OPCODE)
    return rpc::RPC_UNHANDLED_OPCODE;

  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    __ubsan_offload_report R;
    internal_memcpy(&R, Buffer->data, sizeof(R));
    PrintOffloadReport(R);
  });
  return rpc::RPC_SUCCESS;
}

// The OpenMP offloading runtime has its own RPC server we can use if present.
bool tryOffload() {
  if (Offload)
    return true;
  using RegisterFn = void (*)(uint32_t (*)(void *, uint32_t));
  auto Register = reinterpret_cast<RegisterFn>(
      dlsym(RTLD_DEFAULT, "__tgt_register_rpc_callback"));
  if (!Register)
    return false;
  Register(handleReport);
  Offload = true;
  VReport(1, "%s: device reports through the offload runtime's server\n",
          SanitizerToolName);
  return true;
}

bool executableNeedsRPC(hsa_executable_t Exec, hsa_agent_t Agent, u64 *Addr) {
  return GetHsa().SymbolAddr(Exec, "__llvm_rpc_client", Agent, Addr);
}

void drainDevice(DeviceRpc &D) {
  if (!D.Buffer)
    return;
  rpc::Server Server(D.Ports, D.Buffer);
  while (auto Port = Server.try_open(D.Lanes)) {
    if (handleReport(&*Port, D.Lanes) == rpc::RPC_UNHANDLED_OPCODE)
      VReport(1, "%s: unexpected opcode 0x%x on the report channel\n",
              SanitizerToolName, Port->get_opcode());
  }
}

void drainAll() {
  for (uptr I = 0; I < Devices.size(); ++I)
    drainDevice(Devices[I]);
}

void *ServerLoop(void *) {
  Hsa &H = GetHsa();
  for (;;) {
    if (!atomic_load_relaxed(&Stop))
      H.WaitDoorbell();
    Lock L(&RpcMutex);
    drainAll();
    if (atomic_load_relaxed(&Stop))
      break;
  }
  return nullptr;
}

void plantDoorbell(void *Buffer) {
  Hsa &H = GetHsa();
  auto *Bell = reinterpret_cast<rpc::Doorbell *>(
      static_cast<u8 *>(Buffer) + rpc::Server::doorbell_offset());
  Bell->value = reinterpret_cast<uint64_t *>(H.DoorbellValue);
  Bell->mailbox = reinterpret_cast<uint64_t *>(H.DoorbellMailbox);
  Bell->event_id = H.DoorbellEvent;
}

bool startThread() {
  if (ServerThread)
    return true;
  if (atomic_load_relaxed(&Stop))
    return false;
  ServerThread = internal_start_thread(ServerLoop, nullptr);
  return ServerThread;
}

bool initDevice(uptr Ordinal, hsa_agent_t Agent) {
  while (Devices.size() < GetHsa().Devices.size()) {
    DeviceRpc Empty = {};
    Devices.push_back(Empty);
  }
  if (Ordinal >= Devices.size())
    return false;

  DeviceRpc &D = Devices[Ordinal];
  if (D.Buffer)
    return true;

  GetHsa().RpcInfo(Agent, &D.Lanes, &D.Ports);
  if (D.Ports > rpc::MAX_PORT_COUNT)
    D.Ports = rpc::MAX_PORT_COUNT;

  const uptr Bytes = rpc::Server::allocation_size(D.Lanes, D.Ports);
  void *Buffer = nullptr;
  if (!GetHsa().AllocFineGrained(Bytes, &Buffer) || !Buffer)
    return false;
  internal_memset(Buffer, 0, Bytes);
  plantDoorbell(Buffer);
  D.Buffer = Buffer;
  VReport(1,
          "%s: serving device UBSan reports on GPU %zu, %u ports, %u lanes\n",
          SanitizerToolName, Ordinal, D.Ports, D.Lanes);
  return true;
}

} // namespace

void FlushRpc() {
  Lock L(&RpcMutex);
  if (Offload)
    return;
  drainAll();
}

// Creates a dedicated user thread to run the RPC server if needed by the
// executable. Each device maintains a separate buffer.
void StartRpc(hsa_executable_t Exec) {
  Lock Rpc(&RpcMutex);
  if (!GetHsa().Ready() || atomic_load_relaxed(&Stop))
    return;
  if (tryOffload())
    return;

  Lock Dev(&UbsanOffloadMutex);
  Hsa &Runtime = GetHsa();
  if (!Runtime.Ready() || atomic_load_relaxed(&Stop))
    return;
  for (uptr I = 0; I < Runtime.Devices.size(); ++I) {
    u64 Addr = 0;
    if (!executableNeedsRPC(Exec, Runtime.Devices[I], &Addr))
      continue;
    if (!initDevice(I, Runtime.Devices[I]) || !startThread()) {
      Report("ERROR: %s: failed to start device UBSan RPC\n",
             SanitizerToolName);
      Die();
    }

    // Copies the RPC channel to the corresponding client symbol on the device.
    DeviceRpc &D = Devices[I];
    rpc::Client Client(D.Ports, D.Buffer);
    if (!Runtime.Copy(reinterpret_cast<void *>(Addr), &Client,
                      sizeof(Client))) {
      Report("ERROR: %s: failed to install device UBSan RPC client\n",
             SanitizerToolName);
      Die();
    }
  }
}

// Shuts down the RPC server thread and clears any pending work.
void StopRpc() {
  void *Join = nullptr;
  {
    Lock L(&RpcMutex);
    Offload = false;
    atomic_store_relaxed(&Stop, 1);
    if (ServerThread) {
      if (GetHsa().Doorbell.handle)
        GetHsa().KickDoorbell();
      Join = ServerThread;
      ServerThread = nullptr;
    }
  }
  if (Join)
    internal_join_thread(Join);

  Lock L(&RpcMutex);
  if (ServerThread)
    return;
  for (uptr I = 0; I < Devices.size(); ++I) {
    if (Devices[I].Buffer)
      GetHsa().Free(Devices[I].Buffer);
  }
  Devices.clear();
  atomic_store_relaxed(&Stop, 0);
}

} // namespace __ubsan
