//===- reoptimize.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code required to load the rest of the ELF-on-*IX runtime.
//
//===----------------------------------------------------------------------===//

#include "jit_dispatch.h"
#include "wrapper_function_utils.h"

using namespace orc_rt;

ORC_RT_JIT_DISPATCH_TAG(__orc_rt_reoptimize_tag)

ORC_RT_INTERFACE void __orc_rt_reoptimize(uint64_t MUID, uint32_t CurVersion) {
  if (auto Err = WrapperFunction<void(uint64_t, uint32_t)>::call(
          JITDispatch(&__orc_rt_reoptimize_tag), MUID, CurVersion)) {
    __orc_rt_log_error(toString(std::move(Err)).c_str());
    // FIXME: Should we abort here? Depending on the error we can't guarantee
    //        that the JIT'd code is in a consistent state.
  }
}
