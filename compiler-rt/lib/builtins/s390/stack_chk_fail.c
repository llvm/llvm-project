//===-- stack_chk_fail.c - Implement __stack_chk_fail ---------------------===//
//
// Part of the LLVM Project,under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements __stack_chk_fail for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#ifdef __MVS__
__attribute__((no_stack_protector)) void __stack_chk_fail(void) {
  __asm__ volatile(
      /* Load LAA */
      "  llgt   6,1208\n"
      /* Load LCA */
      "  lg     6,88(6)\n"
      /* Load CAA */
      "  lg     6,8(6)\n"
      /* Load vector */
      "  lg     6,640(6)\n"
      /* Load __CEL4SFCR */
      "  lg     6,32(6)\n"
      /* Branch to __CEL4SFCR */
      "  basr   7,6\n"
      /* NOPR 0 */
      "  bcr    0,0\n"
      :
      :
      : "r6", "r7");
}
#endif
