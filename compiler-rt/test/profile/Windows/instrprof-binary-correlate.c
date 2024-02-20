// REQUIRES: target={{.*windows-msvc.*}}
// REQUIRES: lld-available

// Test binary correlate with build id.

// Three binaries are built with build id.
// RUN: rm -rf %t.dir && split-file %s %t.dir
// RUN: %clang_profgen %t.dir/main.c %t.dir/foo.c %t.dir/bar.c -o %t.dir/main.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.dir/main.exe
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw
// RUN: llvm-profdata show --all-functions --counts %t.normal.profdata > %t.normal.profdata.show

// RUN: %clang_profgen -mllvm -profile-correlate=binary %t.dir/foo.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/foo.dll
// RUN: %clang_profgen -mllvm -profile-correlate=binary %t.dir/bar.c -fuse-ld=lld -Wl,-build-id -Wl,-dll -o %t.dir/bar.dll
// RUN: %clang_profgen -mllvm -profile-correlate=binary %t.dir/main.c -fuse-ld=lld -Wl,-build-id %t.dir/foo.lib %t.dir/bar.lib -o %t.dir/main.exe
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t.dir/main.exe
// RUN: llvm-profdata merge -o %t.profdata %t.proflite --binary-file=%t.dir/foo.dll,%t.dir/bar.dll,%t.dir/main.exe
// RUN: llvm-profdata show --all-functions --counts %t.profdata > %t.profdata.show
// RUN: diff %t.normal.profdata.show %t.profdata.show

// One binary is built without build id.
// RUN: %clang_profgen -mllvm -profile-correlate=binary %t.dir/main.c -fuse-ld=lld %t.dir/foo.lib %t.dir/bar.lib -o %t.dir/main.exe
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t.dir/main.exe
// RUN: llvm-profdata merge -o %t.profdata %t.proflite --binary-file=%t.dir/foo.dll,%t.dir/bar.dll,%t.dir/main.exe
// RUN: llvm-profdata show --all-functions --counts %t.profdata > %t.profdata.show
// RUN: diff %t.normal.profdata.show %t.profdata.show

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
