//===- emisssary_device_utils.cpp - utils for Emissary APIs ------- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device functions emitted by clang/lib/CodeGen/EmitEmissaryExec.cpp
//
//===----------------------------------------------------------------------===//

// #include "Allocator.h"
// #include "Configuration.h"
// #include "DeviceTypes.h"
#include "EmissaryIds.h"
#include "rpc_client.h"
#include "shared/rpc.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"

extern "C" {

#ifdef __NVPTX__
[[gnu::leaf]] void *malloc(size_t Size);
[[gnu::leaf]] void free(void *Ptr);
#endif

/// static auto null_string = "(null)";

// namespace LIBC_NAMESPACE_DECL{
// namespace rpc {

// The clang compiler will generate calls to __strlen_max when string length
// is not compile time constant.
uint32_t __strlen_max(const char *instr, uint32_t maxstrlen) {
  if (instr == 0) // encountered a null pointer to string
    return 0;
  for (uint32_t i = 0; i < maxstrlen; i++)
    if (instr[i] == (char)0)
      return (uint32_t)(i + 1);
  return maxstrlen;
}

void *__llvm_emissary_premalloc(uint32_t sz) {
#ifdef __NVPTX__
  return malloc((size_t)sz);
#else
  return LIBC_NAMESPACE::malloc((size_t)sz);
#endif
}
unsigned long long __llvm_emissary_rpc(uint32_t sz32, void *bufdata) {
  rpc::Client::Port Port = LIBC_NAMESPACE::rpc::client.open<OFFLOAD_EMISSARY>();
  Port.send_n(bufdata, (size_t)sz32);
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
#ifdef __NVPTX__
  free(bufdata);
#else
  LIBC_NAMESPACE::free(bufdata);
#endif
  return Ret;
}

// This is for emissary APIs that require d2h or h2d memory transfers.
unsigned long long __llvm_emissary_rpc_dm(uint32_t sz32, void *bufdata) {
  rpc::Client::Port Port =
      LIBC_NAMESPACE::rpc::client.open<OFFLOAD_EMISSARY_DM>();
  Port.send_n(bufdata, (size_t)sz32);
  char *data = (char *)bufdata;
  uint32_t *int32_data = (uint32_t *)data;
  uint32_t NumArgs = int32_data[1];
  char *keyptr = data + (2 * sizeof(int));
  char *argptr = keyptr + (NumArgs * sizeof(int));
  if (((size_t)argptr) % (size_t)8)
    argptr += 4; // argptr must be aligned
  uint64_t arg1 = *(uint64_t *)argptr;
  uint32_t NumSendXfers = (unsigned int)((arg1 >> 16) & 0xFFFF);
  uint32_t NumRecvXfers = (unsigned int)((arg1) & 0xFFFF);
  // Skip by arg1 and process Send and Recv Xfers if any
  argptr += sizeof(uint64_t);
  for (uint32_t idx = 0; idx < NumSendXfers; idx++) {
    void *D2Hdata = (void *)*((uint64_t *)argptr);
    argptr += sizeof(void *);
    size_t D2Hsize = ((size_t)*((size_t *)argptr) & 0x00000000FFFFFFFF);
    argptr += sizeof(size_t);
    Port.send_n(D2Hdata, D2Hsize);
  }
  for (uint32_t idx = 0; idx < NumRecvXfers; idx++) {
    void *H2Ddata = (void *)*((uint64_t *)argptr);
    argptr += sizeof(void *);
    argptr += sizeof(size_t);
    uint64_t recv_size;
    void *buf = nullptr;
    Port.recv_n(&buf, &recv_size,
                [&](uint64_t) { return reinterpret_cast<void *>(H2Ddata); });
  }
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
#ifdef __NVPTX__
  free(bufdata);
#else
  LIBC_NAMESPACE::free(bufdata);
#endif
  return Ret;
}

//} // end namespace rpc
//} // end  namespace LIBC_NAMESPACE_DECL
} // end extern "C"
