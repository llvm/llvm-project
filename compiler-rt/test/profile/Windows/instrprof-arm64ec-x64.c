// REQUIRES: target={{arm64ec-.*windows-msvc.*}}
//
// The x64 object contributes an indexed record. The Arm64EC caller and
// runtime must be able to link it and write both sets of counts.
// RUN: %clang_profgen --target=x86_64-pc-windows-msvc -DX64 -c %s -o %t.x64.obj
// RUN: %clang_profgen %s %t.x64.obj -o %t.exe
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.exe
// RUN: llvm-profdata show --all-functions --counts %t.profraw > %t.txt
// RUN: FileCheck %s --check-prefix=LEAF < %t.txt
// RUN: FileCheck %s --check-prefix=MAIN < %t.txt

#ifdef X64
int leaf(int n) { return n == 1 ? 3 : n; }
#else
int leaf(int);
int main(void) {
  int total = 0;
  for (int i = 0; i != 3; ++i)
    total += leaf(i);
  return total != 5;
}
#endif

// LEAF-LABEL: leaf:
// LEAF:       Function count: 3
// LEAF:       Block counts: [1]
// MAIN-LABEL: main:
// MAIN:       Function count: 1
// MAIN:       Block counts: [3]
