//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-threads

// <stdatomic.h>

// template<class T>
//   using std-atomic = std::atomic<T>;        // exposition only
//
// #define _Atomic(T) std-atomic<T>

// Verify that _Atomic(T) directly uses an alias template but not the std::atomic class template.
// See also https://llvm.org/PR168579.

#include <stdatomic.h>

struct _Atomic(int) x;
// expected-error-re@-1{{alias template '{{.*}}' cannot be referenced with the 'struct' specifier}}
