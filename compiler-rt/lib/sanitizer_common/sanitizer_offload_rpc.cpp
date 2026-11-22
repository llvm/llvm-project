//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host RPC server for device sanitizer reports. One server, many opcode
// handlers. Prefers liboffload's server when present. Otherwise plants a
// client on each executable's empty __llvm_rpc_client, sharing one buffer
// per device.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_offload_rpc.h"

#include <dlfcn.h>
#include <pthread.h>

#include "sanitizer_allocator_internal.h"
#include "sanitizer_atomic.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_offload.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_posix.h"
#include "shared/rpc.h"

namespace __sanitizer {
namespace {

struct DeviceRpc {
  void* Buffer;
  rpc::Server* Server;
  u32 Ports;
  u32 Lanes;
};

struct Rpc {
  hsa_signal_t Signal{};
  u64* SignalValue{};
  u64* SignalMailbox{};
  u32 SignalEvent{};
  InternalMmapVectorNoCtor<DeviceRpc> Slots{};
  InternalMmapVectorNoCtor<Offload::Handler> Handlers{};
  void* Thread{};
  bool Liboffload{};
  bool RegisteredLiboffload{};
  atomic_uint8_t Halt{};
  Mutex Mtx{};
  Mutex HandlerMtx{};
};

Rpc State;

Rpc& GetRpc() { return State; }

u32 Dispatch(Rpc& R, void* PortPtr, u32 Lanes) {
  Lock L(&R.HandlerMtx);
  for (uptr I = 0; I < R.Handlers.size(); ++I) {
    u32 Status = R.Handlers[I](PortPtr, Lanes);
    if (Status != rpc::RPC_UNHANDLED_OPCODE)
      return Status;
  }
  return rpc::RPC_UNHANDLED_OPCODE;
}

bool TryLiboffload(Rpc& R) {
  using RegisterFn = void (*)(u32 (*)(void*, u32));
  auto Register = reinterpret_cast<RegisterFn>(
      dlsym(RTLD_DEFAULT, "__tgt_register_rpc_callback"));
  if (!Register) {
    R.Liboffload = false;
    return false;
  }
  if (!R.RegisteredLiboffload) {
    Register([](void* Port, u32 Lanes) -> u32 {
      return Dispatch(GetRpc(), Port, Lanes);
    });
    R.RegisteredLiboffload = true;
    VReport(1, "%s: device reports through the offload runtime's server\n",
            SanitizerToolName);
  }
  R.Liboffload = true;
  return true;
}

void DrainSlot(DeviceRpc& D) {
  if (!D.Server)
    return;
  while (auto Port = D.Server->try_open(D.Lanes)) {
    if (Dispatch(GetRpc(), &*Port, D.Lanes) == rpc::RPC_UNHANDLED_OPCODE)
      VReport(1, "%s: unexpected opcode 0x%x on the report channel\n",
              SanitizerToolName, Port->get_opcode());
  }
}

void DrainAll(Rpc& R) {
  for (uptr I = 0; I < R.Slots.size(); ++I) DrainSlot(R.Slots[I]);
}

void FailRpc() {
  Report("ERROR: %s: failed to start device RPC\n", SanitizerToolName);
  Die();
}

void PlantSignal(Rpc& R, void* Buffer) {
  auto* Bell = reinterpret_cast<rpc::Doorbell*>(static_cast<u8*>(Buffer) +
                                                rpc::Server::doorbell_offset());
  Bell->value = reinterpret_cast<uint64_t*>(R.SignalValue);
  Bell->mailbox = reinterpret_cast<uint64_t*>(R.SignalMailbox);
  Bell->event_id = R.SignalEvent;
}

}  // namespace

int internal_pthread_create(void* Th, void* Attr, void* (*Callback)(void*),
                            void* Param) {
  return pthread_create(reinterpret_cast<pthread_t*>(Th),
                        reinterpret_cast<const pthread_attr_t*>(Attr), Callback,
                        Param);
}

int internal_pthread_join(void* Th, void** Ret) {
  return pthread_join(reinterpret_cast<pthread_t>(Th), Ret);
}

void* OffloadRpc::ServerLoop(void* Arg) {
  Offload& O = *static_cast<Offload*>(Arg);
  Rpc& R = GetRpc();
  for (;;) {
    if (!atomic_load_relaxed(&R.Halt))
      O.WaitSignal(R.Signal);
    Lock L(&R.Mtx);
    DrainAll(R);
    if (atomic_load_relaxed(&R.Halt))
      break;
  }
  return nullptr;
}

void OffloadRpc::RegisterHandler(Offload::Handler Fn) {
  Rpc& R = GetRpc();
  Lock L(&R.HandlerMtx);
  for (uptr I = 0; I < R.Handlers.size(); ++I)
    if (R.Handlers[I] == Fn)
      return;
  R.Handlers.push_back(Fn);
}

void OffloadRpc::Flush() {
  Rpc& R = GetRpc();
  Lock L(&R.Mtx);
  if (R.Liboffload)
    return;
  DrainAll(R);
}

void OffloadRpc::Start(Offload& O, hsa_executable_t Exec) {
  Rpc& R = GetRpc();
  Lock L(&R.Mtx);
  if (!O.Ready() || atomic_load_relaxed(&R.Halt))
    return;
  if (TryLiboffload(R))
    return;

  if (!R.Signal.handle) {
    if (!O.CreateSignal(&R.Signal))
      FailRpc();
    // Mirror of ROCr amd_signal_t: KFD interrupt slot used to wake the RPC
    // server thread.
    struct AMDSignal {
      int64_t Kind;
      int64_t Value;
      uint64_t EventMailboxPtr;
      uint32_t EventId;
    };
    auto* S = reinterpret_cast<AMDSignal*>(R.Signal.handle);
    R.SignalValue = reinterpret_cast<u64*>(&S->Value);
    R.SignalMailbox = reinterpret_cast<u64*>(S->EventMailboxPtr);
    R.SignalEvent = S->EventId;
  }

  while (R.Slots.size() < O.Devices().size()) {
    DeviceRpc Empty = {};
    R.Slots.push_back(Empty);
  }

  for (uptr I = 0; I < O.Devices().size(); ++I) {
    const Offload::Device& D = O.Devices()[I];
    u64 Addr = 0;
    if (!O.Lookup(Exec, "__llvm_rpc_client", D.Agent, &Addr))
      continue;

    DeviceRpc& Slot = R.Slots[I];
    if (!Slot.Buffer) {
      Slot.Lanes = D.Lanes;
      Slot.Ports = D.MaxWaves;
      if (Slot.Ports > rpc::MAX_PORT_COUNT)
        Slot.Ports = rpc::MAX_PORT_COUNT;
      const uptr Bytes = rpc::Server::allocation_size(Slot.Lanes, Slot.Ports);
      void* Buffer = nullptr;
      if (!O.Alloc(O.Host(), Bytes, &Buffer) || !Buffer)
        FailRpc();
      internal_memset(Buffer, 0, Bytes);
      PlantSignal(R, Buffer);
      Slot.Buffer = Buffer;
      Slot.Server = new (InternalAlloc(sizeof(rpc::Server)))
          rpc::Server(Slot.Ports, Slot.Buffer);
      VReport(1, "%s: serving device reports on GPU %zu, %u ports, %u lanes\n",
              SanitizerToolName, I, Slot.Ports, Slot.Lanes);
    }
    auto* Client = new (InternalAlloc(sizeof(rpc::Client)))
        rpc::Client(Slot.Ports, Slot.Buffer);
    bool Installed =
        O.Copy(reinterpret_cast<void*>(Addr), Client, sizeof(*Client));
    Client->~Client();
    InternalFree(Client);
    if (!Installed)
      FailRpc();

    if (!R.Thread) {
      if (atomic_load_relaxed(&R.Halt) ||
          !(R.Thread = internal_start_thread(ServerLoop, &O)))
        FailRpc();
    }
  }
}

void OffloadRpc::Stop(Offload& O) {
  Rpc& R = GetRpc();
  void* Join = nullptr;
  {
    Lock L(&R.Mtx);
    R.Liboffload = false;
    atomic_store_relaxed(&R.Halt, 1);
    if (R.Thread) {
      if (R.Signal.handle)
        O.StoreSignal(R.Signal);
      Join = R.Thread;
      R.Thread = nullptr;
    }
  }
  if (Join)
    internal_join_thread(Join);

  Lock L(&R.Mtx);
  if (R.Thread)
    return;
  for (uptr I = 0; I < R.Slots.size(); ++I) {
    if (R.Slots[I].Server) {
      R.Slots[I].Server->~Server();
      InternalFree(R.Slots[I].Server);
    }
    if (R.Slots[I].Buffer)
      O.Free(R.Slots[I].Buffer);
  }
  R.Slots.clear();
  if (R.Signal.handle) {
    O.DestroySignal(R.Signal);
    R.Signal = {};
    R.SignalValue = nullptr;
    R.SignalMailbox = nullptr;
    R.SignalEvent = 0;
  }
  atomic_store_relaxed(&R.Halt, 0);
}

}  // namespace __sanitizer
