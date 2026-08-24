//===- emissary_device_utils.cpp - utils for Emissary APIs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device functions emitted by clang/lib/CodeGen/CGEmitEmissaryExec.cpp
//
//===----------------------------------------------------------------------===//

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

// The clang compiler will generate calls to __strlen_max when string length
// is not compile time constant.
uint32_t __strlen_max(const char *InStr, uint32_t MaxStrLen) {
  if (InStr == 0) // encountered a null pointer to string
    return 0;
  for (uint32_t I = 0; I < MaxStrLen; I++)
    if (InStr[I] == (char)0)
      return (uint32_t)(I + 1);
  return MaxStrLen;
}

void *__llvm_emissary_premalloc(uint32_t Sz) {
#ifdef __NVPTX__
  return malloc((size_t)Sz);
#else
  return LIBC_NAMESPACE::malloc((size_t)Sz);
#endif
}
unsigned long long __llvm_emissary_rpc(uint32_t Sz32, void *BufData) {
  rpc::Client::Port Port = LIBC_NAMESPACE::rpc::client.open<OFFLOAD_EMISSARY>();
  Port.send_n(BufData, (size_t)Sz32);
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
#ifdef __NVPTX__
  free(BufData);
#else
  LIBC_NAMESPACE::free(BufData);
#endif
  return Ret;
}

// This is for emissary APIs that require d2h or h2d memory transfers.
unsigned long long __llvm_emissary_rpc_dm(uint32_t Sz32, void *BufData) {
  rpc::Client::Port Port =
      LIBC_NAMESPACE::rpc::client.open<OFFLOAD_EMISSARY_DM>();
  Port.send_n(BufData, (size_t)Sz32);
  char *Data = (char *)BufData;
  uint32_t *Int32Data = (uint32_t *)Data;
  uint32_t NumArgs = Int32Data[1];
  char *KeyPtr = Data + (2 * sizeof(int));
  char *ArgPtr = KeyPtr + (NumArgs * sizeof(int));
  if (((size_t)ArgPtr) % (size_t)8)
    ArgPtr += 4; // ArgPtr must be aligned
  uint64_t Arg1 = *(uint64_t *)ArgPtr;
  uint32_t NumSendXfers = (unsigned int)((Arg1 >> 16) & 0xFFFF);
  uint32_t NumRecvXfers = (unsigned int)((Arg1) & 0xFFFF);
  // Skip by Arg1 and process Send and Recv Xfers if any
  ArgPtr += sizeof(uint64_t);
  for (uint32_t idx = 0; idx < NumSendXfers; idx++) {
    void *D2HData = (void *)*((uint64_t *)ArgPtr);
    ArgPtr += sizeof(void *);
    size_t D2HSize = ((size_t)*((size_t *)ArgPtr) & 0x00000000FFFFFFFF);
    ArgPtr += sizeof(size_t);
    Port.send_n(D2HData, D2HSize);
  }
  for (uint32_t idx = 0; idx < NumRecvXfers; idx++) {
    void *H2DData = (void *)*((uint64_t *)ArgPtr);
    ArgPtr += sizeof(void *);
    ArgPtr += sizeof(size_t);
    uint64_t RecvSize;
    void *Buf = nullptr;
    Port.recv_n(&Buf, &RecvSize,
                [&](uint64_t) { return reinterpret_cast<void *>(H2DData); });
  }
  unsigned long long Ret;
  Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
    Ret = static_cast<unsigned long long>(Buffer->data[0]);
  });
#ifdef __NVPTX__
  free(BufData);
#else
  LIBC_NAMESPACE::free(BufData);
#endif
  return Ret;
}
} // end extern "C"
