// RUN: %clang -target x86_64-linux-gnu -gdwarf-2 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target x86_64-linux-gnu -gdwarf-3 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER3
// RUN: %clang -target x86_64-linux-gnu -gdwarf-4 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-linux-gnu -gdwarf-5 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER5
// RUN: %clang -target x86_64-linux-gnu -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-linux-gnu -gdwarf -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-apple-darwin -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-unknown-openbsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-unknown-freebsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target i386-pc-solaris -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
//
// Test what -gcodeview and -gdwarf do on Windows.
// RUN: %clang -target i686-pc-windows-msvc -gcodeview -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NODWARF --check-prefix=CODEVIEW
// RUN: %clang -target i686-pc-windows-msvc -gdwarf -gcodeview -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4 --check-prefix=CODEVIEW
int main (void) {
  return 0;
}

// VER2: !{i32 2, !"Dwarf Version", i32 2}
// VER3: !{i32 2, !"Dwarf Version", i32 3}
// VER4: !{i32 2, !"Dwarf Version", i32 4}
// VER5: !{i32 2, !"Dwarf Version", i32 5}

// NODWARF-NOT: !"Dwarf Version"
// CODEVIEW: !{i32 2, !"CodeView", i32 1}
// NODWARF-NOT: !"Dwarf Version"
