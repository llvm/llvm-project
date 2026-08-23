// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS
  // [saferecl.hp.base], class template hazard_pointer_obj_base
  using std::hazard_pointer_obj_base;

  // [saferecl.hp.holder], class hazard_pointer
  using std::hazard_pointer;

  // [saferecl.hp.holder.nonmem], non-member functions
  using std::make_hazard_pointer;
  using std::swap;
#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS
} // namespace std
