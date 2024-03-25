// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
  using ::max_align_t;
  using ::nullptr_t;
  using ::ptrdiff_t;
  using ::size_t;

  // [support.c.headers]/1
  // ...  placed within the global namespace scope, except for ... the
  // declaration of std::byte ([cstddef.syn]), and the functions and
  // function templates described in [support.types.byteops]. ...

  // [support.types.byteops], byte type operations
} // export
