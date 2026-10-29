//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for syscall
///
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/compat/syscall.h"
#include "test/UnitTest/Test.h"

#include <sys/syscall.h>

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#include <asm/prctl.h>
#endif

// unistd.h states syscall as a macro; the function is what is under test.
#undef syscall

TEST(LlvmLibcSyscallTest, ReturnsWhatTheKernelDoes) {
  ASSERT_EQ(LIBC_NAMESPACE::syscall(SYS_getpid),
            LIBC_NAMESPACE::syscall_impl<long>(SYS_getpid));
}

#ifdef LIBC_TARGET_ARCH_IS_X86_64
// Wine's signal handlers find %fs pointing at the 32 bit TEB and call syscall
// to move it back, so syscall starts with one thread pointer and ends with
// another.
TEST(LlvmLibcSyscallTest, MayMoveTheThreadPointer) {
  unsigned long own = 0;
  ASSERT_EQ(LIBC_NAMESPACE::syscall(SYS_arch_prctl, ARCH_GET_FS, &own), 0L);

  // A block whose canary, at 0x28, is not the thread's.
  alignas(64) static unsigned long other[8];
  other[0] = reinterpret_cast<unsigned long>(other);
  other[5] = ~reinterpret_cast<unsigned long *>(own)[5];

  LIBC_NAMESPACE::syscall_impl<long>(SYS_arch_prctl, ARCH_SET_FS, other);
  long result = LIBC_NAMESPACE::syscall(SYS_arch_prctl, ARCH_SET_FS, own);
  ASSERT_EQ(result, 0L);

  unsigned long now = 0;
  ASSERT_EQ(LIBC_NAMESPACE::syscall(SYS_arch_prctl, ARCH_GET_FS, &now), 0L);
  EXPECT_EQ(now, own);
}
#endif
