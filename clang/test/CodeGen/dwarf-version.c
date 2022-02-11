// RUN: %clang -target x86_64-linux-gnu -gdwarf-2 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target x86_64-linux-gnu -gdwarf-3 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER3
// RUN: %clang -target x86_64-linux-gnu -gdwarf-4 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-linux-gnu -gdwarf-5 -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER5
// RUN: %clang -target x86_64-linux-gnu -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER5
// RUN: %clang -target x86_64-linux-gnu -gdwarf -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER5

// The -isysroot is used as a hack to avoid LIT messing with the SDKROOT
// environment variable which indirecty overrides the version in the target
// triple used here.
// RUN: %clang -target x86_64-apple-macosx10.11 -g -S -emit-llvm -o - %s -isysroot %t | FileCheck %s --check-prefix=VER4
// RUN: %clang -target x86_64-apple-darwin14 -g -S -emit-llvm -o - %s -isysroot %t | FileCheck %s --check-prefix=VER2

// RUN: %clang -target powerpc-unknown-openbsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-unknown-freebsd -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target i386-pc-solaris -g -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2
// RUN: %clang -target i386-pc-solaris -gdwarf -S -emit-llvm -o - %s | FileCheck %s --check-prefix=VER2

// Check which debug info formats we use on Windows. By default, in an MSVC
// environment, we should use codeview. You can enable dwarf, which implicitly
// disables codeview, of you can explicitly ask for both if you don't know how
// the app will be debugged.
//     Default is codeview.
// RUN: %clang -target i686-pc-windows-msvc -g -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=NODWARF,CODEVIEW
//     Explicitly request codeview.
// RUN: %clang -target i686-pc-windows-msvc -gcodeview -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=NODWARF,CODEVIEW
//     Explicitly request DWARF.
// RUN: %clang -target i686-pc-windows-msvc -gdwarf -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=VER4,NOCODEVIEW
//     Explicitly request both.
// RUN: %clang -target i686-pc-windows-msvc -gdwarf -gcodeview -S -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefixes=VER4,CODEVIEW

// Check what version of dwarf is used for MinGW targets.
// RUN: %clang -target i686-pc-windows-gnu -g -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=VER4

// RUN: %clang -target powerpc-ibm-aix-xcoff -g -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=VER3
// RUN: %clang -target powerpc-ibm-aix-xcoff -gdwarf-2 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=VER2
// RUN: %clang -target powerpc-ibm-aix-xcoff -gdwarf-3 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=VER3
// RUN: %clang -target powerpc-ibm-aix-xcoff -gdwarf-4 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=VER4
// RUN: %clang -target powerpc-ibm-aix-xcoff -gdwarf-5 -S -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=VER5

int main (void) {
  return 0;
}

// NOCODEVIEW-NOT: !"CodeView"

// VER2: !{i32 7, !"Dwarf Version", i32 2}
// VER3: !{i32 7, !"Dwarf Version", i32 3}
// VER4: !{i32 7, !"Dwarf Version", i32 4}
// VER5: !{i32 7, !"Dwarf Version", i32 5}

// NODWARF-NOT: !"Dwarf Version"
// CODEVIEW: !{i32 2, !"CodeView", i32 1}
// NOCODEVIEW-NOT: !"CodeView"
// NODWARF-NOT: !"Dwarf Version"
