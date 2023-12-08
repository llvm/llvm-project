// REQUIRES: lld-available

// FIXME: Investigate and fix.
// XFAIL: powerpc64-target-arch

// RUN: %clang_profgen -fcoverage-mapping -c %s -o %t0.o
// RUN: %clang_profgen -fcoverage-mapping -c %s -DOBJ_1 -o %t1.o
// RUN: %clang_profgen -fcoverage-mapping -c %s -DOBJ_2 -o %t2.o

/// An external symbol can override a weak external symbol.
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1_NOGC
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,--gc-sections %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1_NOGC

// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2_NOGC
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,--gc-sections %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2_GC

/// Repeat the above tests with -ffunction-sections.
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -o %t0.o
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -DOBJ_1 -o %t1.o
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -DOBJ_2 -o %t2.o

// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1_NOGC
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,--gc-sections %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1_GC

// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2_NOGC
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,--gc-sections %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2_GC

// CHECK1: strong
// CHECK1: strong

/// __profc__Z4weakv in %t1.o is local and has a zero value.
/// Without GC it takes a duplicate entry.
// PROFILE1_NOGC:      _Z4weakv:
// PROFILE1_NOGC-NEXT:    Hash:
// PROFILE1_NOGC-NEXT:    Counters: 1
// PROFILE1_NOGC-NEXT:    Function count: 0
// PROFILE1_NOGC:      _Z4weakv:
// PROFILE1_NOGC-NEXT:    Hash:
// PROFILE1_NOGC-NEXT:    Counters: 1
// PROFILE1_NOGC-NEXT:    Function count: 2

// PROFILE1_GC:      _Z4weakv:
// PROFILE1_GC-NEXT:    Hash:
// PROFILE1_GC-NEXT:    Counters: 1
// PROFILE1_GC-NEXT:    Function count: 2
// PROFILE1_GC-NOT:  _Z4weakv:

// CHECK2: weak
// CHECK2: weak

/// __profc__Z4weakv in %t2.o is weak and resolves to the value of %t0.o's copy.
/// Without GC it takes a duplicate entry.
// PROFILE2_NOGC:      _Z4weakv:
// PROFILE2_NOGC-NEXT:    Hash:
// PROFILE2_NOGC-NEXT:    Counters: 1
// PROFILE2_NOGC-NEXT:    Function count: 2
// PROFILE2_NOGC:      _Z4weakv:
// PROFILE2_NOGC-NEXT:    Hash:
// PROFILE2_NOGC-NEXT:    Counters: 1
// PROFILE2_NOGC-NEXT:    Function count: 2

// PROFILE2_GC:      _Z4weakv:
// PROFILE2_GC-NEXT:    Hash:
// PROFILE2_GC-NEXT:    Counters: 1
// PROFILE2_GC-NEXT:    Function count: 2
// PROFILE2_GC-NOT:  _Z4weakv:

#ifdef OBJ_1
#include <stdio.h>

void weak() { puts("strong"); }
void foo() { weak(); }

#elif defined(OBJ_2)
#include <stdio.h>

__attribute__((weak)) void weak() { puts("unreachable"); }
void foo() { weak(); }

#else
#include <stdio.h>

__attribute__((weak)) void weak() { puts("weak"); }
void foo();

int main() {
  foo();
  weak();
}
#endif
