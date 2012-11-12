// General tests that the header search paths detected by the driver and passed
// to CC1 are sane.
//
// Test a very broken version of multiarch that shipped in Ubuntu 11.04.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/ubuntu_11.04_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-11-04 %s
// CHECK-UBUNTU-11-04: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-UBUNTU-11-04: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/i686-linux-gnu"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/backward"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-11-04: "-internal-isystem" "{{.*}}/lib{{(64|32)?}}/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Thoroughly exercise the Debian multiarch environment.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i686-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86 %s
// CHECK-DEBIAN-X86: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-X86: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5/i686-linux-gnu"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-X86: "-internal-isystem" "{{.*}}/lib{{(64|32)?}}/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/i386-linux-gnu"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86-64 %s
// CHECK-DEBIAN-X86-64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-X86-64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5/x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "{{.*}}/lib{{(64|32)?}}/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target powerpc-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC %s
// CHECK-DEBIAN-PPC: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-PPC: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5/powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-PPC: "-internal-isystem" "{{.*}}/lib{{(64|32)?}}/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target powerpc64-linux-gnu \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC64 %s
// CHECK-DEBIAN-PPC64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-PPC64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5/powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "{{.*}}/lib{{(64|32)?}}/clang/{{[0-9]\.[0-9]}}/include"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
