//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XFAIL: libcpp-has-no-threads

#ifdef _LIBCPP_HAS_NO_THREADS
#error This should be XFAIL'd for the purpose of detecting that the LIT feature\
   'libcpp-has-no-threads' is available iff _LIBCPP_HAS_NO_THREADS is defined
#endif

int main()
{
}
