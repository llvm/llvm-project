//===- common.h - Common utilities for the ORC runtime ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the ORC runtime support library.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_COMMON_H
#define ORC_RT_COMMON_H

#include "c_api.h"
#include <type_traits>

/// Opaque struct for external symbols.
struct __orc_rt_Opaque {};

/// Error reporting function.
extern "C" void __orc_rt_log_error(const char *ErrMsg);

/// Context object for dispatching calls to the JIT object.
///
/// This is declared for use by the runtime, but should be implemented in the
/// executor or provided by a definition added to the JIT before the runtime
/// is loaded.
extern "C" __orc_rt_Opaque __orc_rt_jit_dispatch_ctx
  __attribute__((weak_import));

/// For dispatching calls to the JIT object.
///
/// This is declared for use by the runtime, but should be implemented in the
/// executor or provided by a definition added to the JIT before the runtime
/// is loaded.
extern "C" __orc_rt_CWrapperFunctionResult
__orc_rt_jit_dispatch(__orc_rt_Opaque *DispatchCtx, const void *FnTag,
                      const char *Data, size_t Size)
  __attribute__((weak_import));

namespace __orc_rt {

/// Must be kept in sync with JITSymbol.h
using JITTargetAddress = uint64_t;

/// Cast from JITTargetAddress to pointer.
template <typename T> T jitTargetAddressToPointer(JITTargetAddress Addr) {
  static_assert(std::is_pointer<T>::value, "T must be a pointer type");
  return reinterpret_cast<T>(static_cast<uintptr_t>(Addr));
}

/// Cast from pointer to JITTargetAddress.
template <typename T> JITTargetAddress pointerToJITTargetAddress(T *Ptr) {
  return static_cast<JITTargetAddress>(reinterpret_cast<uintptr_t>(Ptr));
}

} // end namespace __orc_rt

#endif // ORC_RT_COMMON_H
