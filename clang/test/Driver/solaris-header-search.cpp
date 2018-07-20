// Test that the C++ headers are found on Solaris with gcc toolchain detection
//
// Sparc, 32bit
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 --stdlib=platform \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK_SOLARIS_SPARC %s
// CHECK_SOLARIS_SPARC: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK_SOLARIS_SPARC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK_SOLARIS_SPARC-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../../include/c++/4.8.2"
// CHECK_SOLARIS_SPARC-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../../include/c++/4.8.2/sparc-sun-solaris2.11"

// Sparc, 64bit
// RUN: %clang -no-canonical-prefixes -m64 %s -### -fsyntax-only 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 --stdlib=platform \
// RUN:     --sysroot=%S/Inputs/solaris_sparc_tree \
// RUN:   | FileCheck --check-prefix=CHECK_SOLARIS_SPARC64 %s
// CHECK_SOLARIS_SPARC64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK_SOLARIS_SPARC64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK_SOLARIS_SPARC64-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../../include/c++/4.8.2"
// CHECK_SOLARIS_SPARC64-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2/../../../../include/c++/4.8.2/sparc-sun-solaris2.11/sparcv9"

// Intel, 32bit
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 --stdlib=platform \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK_SOLARIS_X86 %s
// CHECK_SOLARIS_X86: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK_SOLARIS_X86-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK_SOLARIS_X86-SAME: "-internal-isystem" "{{.*}}/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../../include/c++/4.9.4"
// CHECK_SOLARIS_X86-SAME: "-internal-isystem" "{{.*}}/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../../include/c++/4.9.4/i386-pc-solaris2.11"

// Intel, 64bit
// RUN: %clang -no-canonical-prefixes -m64 %s -### -fsyntax-only 2>&1 \
// RUN:     --target=i386-pc-solaris2.11 --stdlib=platform \
// RUN:     --sysroot=%S/Inputs/solaris_x86_tree \
// RUN:   | FileCheck --check-prefix=CHECK_SOLARIS_X64 %s
// CHECK_SOLARIS_X64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK_SOLARIS_X64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK_SOLARIS_X64-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../../include/c++/4.9.4"
// CHECK_SOLARIS_X64-SAME: "-internal-isystem" "[[SYSROOT]]/usr/gcc/4.9/lib/gcc/i386-pc-solaris2.11/4.9.4/../../../../include/c++/4.9.4/i386-pc-solaris2.11/amd64"
