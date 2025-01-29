// REQUIRES: target={{.*windows-msvc.*}}
// REQUIRES: lld-available

// RUN: %clang_profgen -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.out
// RUN: FileCheck %s --check-prefix=NO-BINARY-ID < %t.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw

// RUN: %clang_profgen -fuse-ld=lld -Wl,-build-id -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-RAW-PROF < %t.profraw.out

// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/default_%m.profraw %run %t
// RUN: llvm-profdata show --binary-ids  %t.profdir/default_*.profraw > %t.profraw.out
// RUN: FileCheck %s --check-prefix=ONE-BINARY-ID < %t.profraw.out

// RUN: llvm-profdata merge -o %t.profdata %t.profdir/default_*.profraw
// RUN: llvm-profdata show --binary-ids %t.profdata > %t.profdata.out
// RUN: FileCheck %s --check-prefix=ONE-BINARY-ID < %t.profdata.out

// Test raw profiles with DLLs.
// RUN: rm -rf %t.dir && split-file %s %t.dir
// RUN: %clang_profgen -O2 %t.dir/foo.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/foo.dll
// RUN: %clang_profgen -O2 %t.dir/bar.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/bar.dll
// RUN: %clang_profgen -O2 %t.dir/main.c -fuse-ld=lld -Wl,-build-id %t.dir/foo.lib %t.dir/bar.lib -o %t.dir/main.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.dir/main.exe
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.profraw.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: FileCheck %s --check-prefix=MULTI-BINARY-ID < %t.profraw.out

// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --binary-ids %t.profdata > %t.profdata.out
// RUN: FileCheck %s --check-prefix=MULTI-BINARY-ID < %t.profraw.out

//--- foo.c
__declspec(dllexport) void foo() {}

//--- bar.c
__declspec(dllexport) void bar() {}

//--- main.c
__declspec(dllimport) void foo();
__declspec(dllimport) void bar();
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
// BINARY-ID-RAW-PROF-NEXT: Total number of blocks:
// BINARY-ID-RAW-PROF-NEXT: Total count:
// BINARY-ID-RAW-PROF-NEXT: Binary IDs:
// BINARY-ID-RAW-PROF-NEXT: {{[0-9a-f]+}}

// ONE-BINARY-ID: Instrumentation level: Front-end
// ONE-BINARY-ID-NEXT: Total functions: 3
// ONE-BINARY-ID-NEXT: Maximum function count: 3
// ONE-BINARY-ID-NEXT: Maximum internal block count: 0
// ONE-BINARY-ID-NEXT: Total number of blocks:
// ONE-BINARY-ID-NEXT: Total count:
// ONE-BINARY-ID-NEXT: Binary IDs:
// ONE-BINARY-ID-NEXT: {{[0-9a-f]+}}

// MULTI-BINARY-ID: Instrumentation level: Front-end
// MULTI-BINARY-ID-NEXT: Total functions: 3
// MULTI-BINARY-ID-NEXT: Maximum function count: 1
// MULTI-BINARY-ID-NEXT: Maximum internal block count: 0
// MULTI-BINARY-ID-NEXT: Total number of blocks:
// MULTI-BINARY-ID-NEXT: Total count:
// MULTI-BINARY-ID-NEXT: Binary IDs:
// MULTI-BINARY-ID-NEXT: {{[0-9a-f]+}}
// MULTI-BINARY-ID-NEXT: {{[0-9a-f]+}}
// MULTI-BINARY-ID-NEXT: {{[0-9a-f]+}}
