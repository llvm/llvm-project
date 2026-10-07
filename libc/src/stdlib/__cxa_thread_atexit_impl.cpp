//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of __cxa_thread_atexit_impl
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/__cxa_thread_atexit_impl.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE_DECL {

// The function __cxa_thread_atexit is provided by C++ runtimes like libcxxabi.
// It is used by thread local object runtime to register destructor calls. To
// actually register destructor call with the threading library, it calls
// __cxa_thread_atexit_impl, which is to be provided by the threading library.
// The semantics are very similar to the __cxa_atexit function except for the
// fact that the registered callback is thread specific.
LLVM_LIBC_FUNCTION(int, __cxa_thread_atexit_impl,
                   (AtExitCallback * callback, void* obj, void*)) {
  return add_thread_atexit_callback(callback, obj);
}
}  // namespace LIBC_NAMESPACE_DECL
