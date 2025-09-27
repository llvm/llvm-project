//===- send_value.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"
#include "debug.h"
#include "jit_dispatch.h"
#include "wrapper_function_utils.h"

using namespace orc_rt;

ORC_RT_JIT_DISPATCH_TAG(__orc_rt_SendResultValue_tag)

ORC_RT_INTERFACE orc_rt_WrapperFunctionResult __orc_rt_runDtor(char *ArgData,
                                                               size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddr, SPSExecutorAddr)>::handle(
             ArgData, ArgSize,
             [](ExecutorAddr DtorFn, ExecutorAddr This) {
               DtorFn.toPtr<void (*)(unsigned char *)>()(
                   This.toPtr<unsigned char *>());
               return Error::success();
             })
      .release();
}

ORC_RT_INTERFACE void __orc_rt_SendResultValue(uint64_t ResultId, void *V) {
  Error OptErr = Error::success();
  if (auto Err = WrapperFunction<SPSError(uint64_t, SPSExecutorAddr)>::call(
          JITDispatch(&__orc_rt_SendResultValue_tag), OptErr, ResultId,
          ExecutorAddr::fromPtr(V))) {
    cantFail(std::move(OptErr));
    cantFail(std::move(Err));
  }
  consumeError(std::move(OptErr));
}
