// REQUIRES: lld-available

// REQUIRES: target={{.*windows-msvc.*}}
// RUN: %clang_profgen -fcoverage-mapping -c %s -o %t0.o
// RUN: %clang_profgen -fcoverage-mapping -c %s -DOBJ_1 -o %t1.o
// RUN: %clang_profgen -fcoverage-mapping -c %s -DOBJ_2 -o %t2.o

/// An external symbol can override a weak external symbol.
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-opt:noref %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-opt:ref %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1

/// link.exe does not support weak overridding weak.
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-lldmingw,-opt:ref %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2

/// Repeat the above tests with -ffunction-sections (associative comdat).
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -o %t0.o
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -DOBJ_1 -o %t1.o
// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -c %s -DOBJ_2 -o %t2.o

// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-opt:noref %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1
// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-opt:ref %t0.o %t1.o -o %t1
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t1 | FileCheck %s --check-prefix=CHECK1
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE1

// RUN: %clang_profgen -fcoverage-mapping -fuse-ld=lld -Wl,-lldmingw,-opt:ref %t0.o %t2.o -o %t2
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t2 | FileCheck %s --check-prefix=CHECK2
// RUN: llvm-profdata show %t.profraw --all-functions | FileCheck %s --check-prefix=PROFILE2

// CHECK1: strong
// CHECK1: strong

/// Document the current behavior:
///  __profc_?weak@@YAXXZ in %t1.o is local and has a zero value.
/// Without GC it takes a duplicate entry.
// PROFILE1:      ?weak@@YAXXZ:
// PROFILE1-NEXT:    Hash:
// PROFILE1-NEXT:    Counters: 1
// PROFILE1-NEXT:    Function count: 0
// PROFILE1:      ?weak@@YAXXZ:
// PROFILE1-NEXT:    Hash:
// PROFILE1-NEXT:    Counters: 1
// PROFILE1-NEXT:    Function count: 2

// CHECK2: weak
// CHECK2: weak

/// __profc__Z4weakv in %t2.o is weak and resolves to the value of %t0.o's copy.
/// Without GC it takes a duplicate entry.
// PROFILE2:      ?weak@@YAXXZ:
// PROFILE2-NEXT:    Hash:
// PROFILE2-NEXT:    Counters: 1
// PROFILE2-NEXT:    Function count: 2
// PROFILE2:      ?weak@@YAXXZ:
// PROFILE2-NEXT:    Hash:
// PROFILE2-NEXT:    Counters: 1
// PROFILE2-NEXT:    Function count: 2

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
