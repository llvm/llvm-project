// REQUIRES: linux
// RUN: %clang_profgen -Wl,--build-id=none -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.out
// RUN: FileCheck %s --check-prefix=NO-BINARY-ID < %t.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw

// RUN: %clang_profgen -Wl,--build-id -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-RAW-PROF < %t.profraw.out

// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: llvm-profdata show --binary-ids  %t.profdir/default_*.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-MERGE-PROF < %t.profraw.out

// RUN: llvm-profdata merge -o %t.profdata %t.profraw %t.profraw
// RUN: llvm-profdata show --binary-ids %t.profdata > %t.profdata.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-INDEXED-PROF < %t.profraw.out

void foo() {
}

void bar() {
}

int main() {
  foo();
  bar();
  return 0;
}

// NO-BINARY-ID: Instrumentation level: Front-end
// NO-BINARY-ID-NEXT: Total functions: 3
// NO-BINARY-ID-NEXT: Maximum function count: 1
// NO-BINARY-ID-NEXT: Maximum internal block count: 0
// NO-BINARY-ID-NOT: Binary IDs:

// BINARY-ID-RAW-PROF: Instrumentation level: Front-end
// BINARY-ID-RAW-PROF-NEXT: Total functions: 3
// BINARY-ID-RAW-PROF-NEXT: Maximum function count: 1
// BINARY-ID-RAW-PROF-NEXT: Maximum internal block count: 0
// BINARY-ID-RAW-PROF-NEXT: Binary IDs:
// BINARY-ID-RAW-PROF-NEXT: {{[0-9a-f]+}}

// BINARY-ID-MERGE-PROF: Instrumentation level: Front-end
// BINARY-ID-MERGE-PROF-NEXT: Total functions: 3
// BINARY-ID-MERGE-PROF-NEXT: Maximum function count: 3
// BINARY-ID-MERGE-PROF-NEXT: Maximum internal block count: 0
// BINARY-ID-MERGE-PROF-NEXT: Binary IDs:
// BINARY-ID-MERGE-PROF-NEXT: {{[0-9a-f]+}}

// BINARY-ID-INDEXED-PROF: Instrumentation level: Front-end
// BINARY-ID-INDEXED-PROF-NEXT: Total functions: 3
// BINARY-ID-INDEXED-PROF-NEXT: Maximum function count: 3
// BINARY-ID-INDEXED-PROF-NEXT: Maximum internal block count: 0
// BINARY-ID-INDEXED-PROF-NEXT: Binary IDs:
// BINARY-ID-INDEXED-PROF-NEXT: {{[0-9a-f]+}}
