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
// RUN: FileCheck %s --check-prefix=BINARY-ID-MERGE-PROF < %t.profraw.out

// RUN: llvm-profdata merge -o %t.profdata %t.profdir/default_*.profraw
// RUN: llvm-profdata show --binary-ids %t.profdata > %t.profdata.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-INDEXED-PROF < %t.profdata.out

// Test raw profiles with DLLs.
// RUN: rm -rf %t.dir && split-file %s %t.dir
// RUN: %clang_profgen -O2 %t.dir/foo.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/foo.dll
// RUN: %clang_profgen -O2 %t.dir/bar.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/bar.dll
// RUN: %clang_profgen -O2 %t.dir/main.c -fuse-ld=lld -Wl,-build-id %t.dir/foo.lib %t.dir/bar.lib -o %t.dir/main.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.dir/main.exe
// RUN: llvm-profdata show --binary-ids %t.profraw > %t.profraw.out
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: FileCheck %s --check-prefix=BINARY-ID-SHARE-RAW-PROF < %t.profraw.out

// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --binary-ids %t.profdata > %t.profdata.out
// RUN: FileCheck %s --check-prefix=BINARY-ID-SHARE-INDEXED-PROF < %t.profraw.out

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

// BINARY-ID-SHARE-RAW-PROF: Instrumentation level: Front-end
// BINARY-ID-SHARE-RAW-PROF-NEXT: Total functions: 3
// BINARY-ID-SHARE-RAW-PROF-NEXT: Maximum function count: 1
// BINARY-ID-SHARE-RAW-PROF-NEXT: Maximum internal block count: 0
// BINARY-ID-SHARE-RAW-PROF-NEXT: Binary IDs:
// BINARY-ID-SHARE-RAW-PROF-NEXT: {{[0-9a-f]+}}
// BINARY-ID-SHARE-RAW-PROF-NEXT: {{[0-9a-f]+}}
// BINARY-ID-SHARE-RAW-PROF-NEXT: {{[0-9a-f]+}}

// BINARY-ID-SHARE-INDEXED-PROF: Instrumentation level: Front-end
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: Total functions: 3
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: Maximum function count: 1
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: Maximum internal block count: 0
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: Binary IDs:
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: {{[0-9a-f]+}}
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: {{[0-9a-f]+}}
// BINARY-ID-SHARE-INDEXED-PROF-NEXT: {{[0-9a-f]+}}
