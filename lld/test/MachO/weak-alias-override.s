# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir -p %t/bin

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/weak.o %t/weak.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/strong_a.o %t/strong_a.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/strong_b.o %t/strong_b.s

# --- Test Case 1: No overrides
# RUN: %lld %t/weak.o -o %t/bin/alone -e _s
# RUN: llvm-nm -am %t/bin/alone | FileCheck --check-prefix=NM_ALONE %s

# NM_ALONE:            [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_ALONE:            [[#P_ADDR]]      (__TEXT,__const) weak external _weak_a
# NM_ALONE:            [[#P_ADDR]]      (__TEXT,__const) weak external _weak_b

# --- Test Case 2: Override weak_a
# RUN: %lld %t/weak.o %t/strong_a.o -o %t/bin/with_a -e _s
# RUN: llvm-nm -am %t/bin/with_a | FileCheck --check-prefix=NM_WITH_A %s
# RUN: llvm-nm -am %t/bin/with_a | FileCheck --check-prefix=NM_WITH_A_BAD %s

# NM_WITH_A:           [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_A:           [[#%x, A_ADDR:]] (__TEXT,__const) external _strong_a
# NM_WITH_A:           [[#A_ADDR]]      (__TEXT,__const) external _weak_a
# NM_WITH_A:           [[#P_ADDR]]      (__TEXT,__const) weak external _weak_b

# --- Addresses of _placeholder_int and _strong_a must not match.
# NM_WITH_A_BAD:       [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_A_BAD-NOT:   [[#P_ADDR]]      (__TEXT,__const) external _strong_a

# --- Test Case 3: Override weak_b
# RUN: %lld %t/weak.o %t/strong_b.o -o %t/bin/with_b -e _s
# RUN: llvm-nm -am %t/bin/with_b | FileCheck --check-prefix=NM_WITH_B %s
# RUN: llvm-nm -am %t/bin/with_b | FileCheck --check-prefix=NM_WITH_B_BAD %s

# NM_WITH_B:           [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_B:           [[#%x, B_ADDR:]] (__TEXT,__const) external _strong_b
# NM_WITH_B:           [[#P_ADDR]]      (__TEXT,__const) weak external _weak_a
# NM_WITH_B:           [[#B_ADDR]]      (__TEXT,__const) external _weak_b

# --- Addresses of _placeholder_int and _strong_a must not match.
# NM_WITH_B_BAD:       [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_B_BAD-NOT:   [[#P_ADDR]]      (__TEXT,__const) external _strong_b

# --- Test Case 4: Override weak_a and weak_b
# RUN: %lld %t/weak.o %t/strong_a.o %t/strong_b.o -o %t/bin/with_ab -e _s
# RUN: llvm-nm -am %t/bin/with_ab | FileCheck --check-prefix=NM_WITH_AB %s
# RUN: llvm-nm -am %t/bin/with_ab | FileCheck --check-prefix=NM_WITH_AB_BAD %s

# NM_WITH_AB:          [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_AB:          [[#%x, A_ADDR:]] (__TEXT,__const) external _strong_a
# NM_WITH_AB:          [[#%x, B_ADDR:]] (__TEXT,__const) external _strong_b
# NM_WITH_AB:          [[#A_ADDR]]      (__TEXT,__const) external _weak_a
# NM_WITH_AB:          [[#B_ADDR]]      (__TEXT,__const) external _weak_b

# --- Addresses of _placeholder_int, _strong_a, and _strong_b must all be distinct
# NM_WITH_AB_BAD:      [[#%x, P_ADDR:]] (__TEXT,__const) weak external _placeholder_int
# NM_WITH_AB_BAD-NOT:  [[#P_ADDR]]      (__TEXT,__const) external _strong_a
# NM_WITH_AB_BAD-NOT:  [[#P_ADDR]]      (__TEXT,__const) external _strong_b

#--- weak.s
.section __TEXT,__const
.globl _placeholder_int
.weak_definition _placeholder_int
_placeholder_int:
 .long 0

.globl _weak_a
.set _weak_a, _placeholder_int
.weak_definition _weak_a

.globl _weak_b
.set _weak_b, _placeholder_int
.weak_definition _weak_b

.globl _s
_s:
 .quad _weak_a
 .quad _weak_b

#--- strong_a.s
.section __TEXT,__const
.globl _strong_a
_strong_a:
 .long 1

.globl _weak_a
_weak_a = _strong_a

#--- strong_b.s
.section __TEXT,__const
.globl _strong_b
_strong_b:
 .long 2

.globl _weak_b
_weak_b = _strong_b
