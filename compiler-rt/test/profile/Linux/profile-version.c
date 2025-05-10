// REQUIRES: linux
// RUN: %clang_profgen -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --profile-version %t.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=RAW-PROF < %t.profraw.out

// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: llvm-profdata show --profile-version %t.profdir/default_*.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=INDEXED-PROF < %t.profraw.out

void foo() {}

void bar() {}

int main() {
  foo();
  bar();
  return 0;
}

// RAW-PROF: Instrumentation level: Front-end
// RAW-PROF-NEXT: Total functions: 3
// RAW-PROF-NEXT: Maximum function count: 1
// RAW-PROF-NEXT: Maximum internal block count: 0
// RAW-PROF-NEXT: Total number of blocks: 3
// RAW-PROF-NEXT: Total count: 3
// RAW-PROF-NEXT: Profile version: {{[0-9]+}}

// INDEXED-PROF: Instrumentation level: Front-end
// INDEXED-PROF-NEXT: Total functions: 3
// INDEXED-PROF-NEXT: Maximum function count: 3
// INDEXED-PROF-NEXT: Maximum internal block count: 0
// INDEXED-PROF-NEXT: Total number of blocks: 3
// INDEXED-PROF-NEXT: Total count: 9
// INDEXED-PROF-NEXT: Profile version: {{[0-9]+}}
