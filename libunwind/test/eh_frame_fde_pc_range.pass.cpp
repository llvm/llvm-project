// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Manually marking the .eh_frame_hdr as DW_EH_PE_omit to make libunwind to do
// the linear search.
// Assuming the begining of the function is at the start of the FDE range.

// clang-format off

// REQUIRES: target={{x86_64-.+-linux-gnu}}
// aarch64,arm have a cross toolchain build(llvm-clang-win-x-aarch64, etc)
// where objdump is not available.

// TODO: Figure out why this fails with Memory Sanitizer.
// XFAIL: msan

// RUN: %{build}
// RUN: objcopy --dump-section .eh_frame_hdr=%t_ehf_hdr.bin %t.exe
// RUN: echo -ne '\xFF' | dd of=%t_ehf_hdr.bin bs=1 seek=2 count=2 conv=notrunc status=none 
// RUN: objcopy --update-section .eh_frame_hdr=%t_ehf_hdr.bin %t.exe
// RUN: %{exec} %t.exe

// clang-format on

#include <assert.h>
#include <libunwind.h>
#include <stdint.h>
#include <stdio.h>
#include <unwind.h>

void f() {
  printf("123\n");
  void *pc = __builtin_return_address(0);
  void *fpc = (void *)&f;
  void *fpc1 = (void *)((uintptr_t)fpc + 1);

  struct dwarf_eh_bases bases;
  const void *fde_pc = _Unwind_Find_FDE(pc, &bases);
  const void *fde_fpc = _Unwind_Find_FDE(fpc, &bases);
  const void *fde_fpc1 = _Unwind_Find_FDE(fpc1, &bases);
  printf("fde_pc = %p\n", fde_pc);
  printf("fde_fpc = %p\n", fde_fpc);
  printf("fde_fpc1 = %p\n", fde_fpc1);
  fflush(stdout);
  assert(fde_pc != NULL);
  assert(fde_fpc != NULL);
  assert(fde_fpc1 != NULL);
  assert(fde_fpc == fde_fpc1);
}

int main() {
  f();
  return 0;
}
