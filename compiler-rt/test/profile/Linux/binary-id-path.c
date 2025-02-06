// REQUIRES: linux
// RUN: split-file %s %t.dir
// RUN: %clang_profgen -Wl,--build-id=sha1 -o %t.dir/foo %t.dir/foo.c
// RUN: %clang_profgen -Wl,--build-id=sha1 -o %t.dir/bar %t.dir/bar.c

// Check that foo and bar have the same signatures.
// RUN: rm -rf %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.profraw %run %t.dir/foo
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.profraw %run %t.dir/bar 2>&1 | FileCheck %s --check-prefix=MERGE-ERROR

// Check that foo and bar have different binary IDs.
// RUN: rm -rf %t.profdir %t.profdata
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%b.profraw %run %t.dir/foo
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%b.profraw %run %t.dir/bar
// RUN: llvm-profdata merge -o %t.profdata %t.profdir
// RUN: llvm-profdata show --binary-ids %t.profdata | FileCheck %s --check-prefix=BINARY-ID

// Check fallback to the default name if binary ID is missing.
// RUN: %clang_profgen -Wl,--build-id=none -o %t.dir/foo %t.dir/foo.c
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%b.profraw %run %t.dir/foo 2>&1 | FileCheck %s --check-prefix=MISSING

// MERGE-ERROR: LLVM Profile Error: Profile Merging of file {{.*}}.profraw failed: File exists

// BINARY-ID: Instrumentation level: Front-end
// BINARY-ID-NEXT: Total functions: 3
// BINARY-ID-NEXT: Maximum function count: 2
// BINARY-ID-NEXT: Maximum internal block count: 0
// BINARY-ID-NEXT: Binary IDs:
// BINARY-ID-NEXT: {{[0-9a-f]+}}
// BINARY-ID-NEXT: {{[0-9a-f]+}}

// MISSING: Unable to get binary ID for filename pattern {{.*}}.profraw. Using the default name.

//--- foo.c
int main(void) { return 0; }
void foo(void) {}

//--- bar.c
int main(void) { return 0; }
void bar(int *a) { *a += 10; }
