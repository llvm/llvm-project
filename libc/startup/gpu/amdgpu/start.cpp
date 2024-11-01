//===-- Implementation of crt for amdgpu ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

extern "C" int main(int argc, char **argv);

extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
_start(int argc, char **argv, int *ret) {
  __atomic_fetch_or(ret, main(argc, argv), __ATOMIC_RELAXED);
}
