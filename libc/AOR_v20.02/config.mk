# Example config.mk
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Subprojects to build
# For now, LLVM-libc project will focus only "math" functions.
SUBS = math # string networking

# Target architecture: aarch64, arm or x86_64
# For now, LLVM-libc project will focus on x86_64 only.
ARCH = x86_64

# Compiler for the target
CC = $(CROSS_COMPILE)gcc
CFLAGS = -std=c99 -pipe -O3
CFLAGS += -Wall -Wno-missing-braces
CFLAGS += -Werror=implicit-function-declaration

# Used for test case generator that is executed on the host
HOST_CC = gcc
HOST_CFLAGS = -std=c99 -O2
HOST_CFLAGS += -Wall -Wno-unused-function

# Enable debug info.
HOST_CFLAGS += -g
CFLAGS += -g

# Optimize the shared libraries on aarch64 assuming they fit in 1M.
#CFLAGS_SHARED = -fPIC -mcmodel=tiny

# Use for cross compilation with gcc.
#CROSS_COMPILE = aarch64-none-linux-gnu-

# Use with cross testing.
#EMULATOR = qemu-aarch64-static
#EMULATOR = sh -c 'scp $$1 user@host:/dir && ssh user@host /dir/"$$@"' --

# Additional flags for subprojects.
math-cflags =
math-ldlibs =
math-ulpflags =
math-testflags =
string-cflags =
networking-cflags =

# Use if mpfr is available on the target for ulp error checking.
#math-ldlibs += -lmpfr -lgmp
#math-cflags += -DUSE_MPFR

# Use with gcc.
math-cflags += -frounding-math -fexcess-precision=standard -fno-stack-protector
math-cflags += -ffp-contract=fast -fno-math-errno

# Use with clang.
#math-cflags += -ffp-contract=fast

# Disable vector math code
#math-cflags += -DWANT_VMATH=0

# Disable fenv checks
#math-ulpflags = -q -f
#math-testflags = -nostatus

# Enable assertion checks.
#networking-cflags += -DWANT_ASSERT

# Avoid auto-vectorization of scalar code and unroll loops
networking-cflags += -O2 -fno-tree-vectorize -funroll-loops
