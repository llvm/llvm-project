//===-- sanitizer_offload_rpc.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_OFFLOAD_RPC_H
#define SANITIZER_OFFLOAD_RPC_H

#include "sanitizer_offload.h"

namespace __sanitizer {

struct OffloadRpc {
  static void Start(Offload& O, hsa_executable_t Exec);
  static void Stop(Offload& O);

 private:
  friend class Offload;

  static void RegisterHandler(Offload::Handler Fn);
  static void Flush();
  static void* ServerLoop(void* Arg);
};

}  // namespace __sanitizer

#endif  // SANITIZER_OFFLOAD_RPC_H
