//===- resolve.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a generic "resolver" function compatible with the
// __orc_rt_reenter function.
//
//===----------------------------------------------------------------------===//

#include "executor_symbol_def.h"
#include "jit_dispatch.h"
#include "wrapper_function_utils.h"

#include <stdio.h>

#define DEBUG_TYPE "resolve"

using namespace orc_rt;

// Declare function tags for functions in the JIT process.
ORC_RT_JIT_DISPATCH_TAG(__orc_rt_resolve_tag)

// FIXME: Make this configurable via an alias.
static void __orc_rt_resolve_fail(void *Caller, const char *ErrMsg) {
  fprintf(stderr, "error resolving implementation for stub %p: %s\n", Caller,
          ErrMsg);
  abort();
}

extern "C" ORC_RT_HIDDEN void *__orc_rt_resolve(void *Caller) {
  Expected<ExecutorSymbolDef> Result((ExecutorSymbolDef()));
  if (auto Err = WrapperFunction<SPSExpected<SPSExecutorSymbolDef>(
          SPSExecutorAddr)>::call(JITDispatch(&__orc_rt_resolve_tag), Result,
                                  ExecutorAddr::fromPtr(Caller))) {
    __orc_rt_resolve_fail(Caller, toString(std::move(Err)).c_str());
    return nullptr; // Unreachable.
  }

  if (!Result) {
    __orc_rt_resolve_fail(Caller, toString(Result.takeError()).c_str());
    return nullptr; // Unreachable.
  }

  return Result->getAddress().toPtr<void *>();
}
