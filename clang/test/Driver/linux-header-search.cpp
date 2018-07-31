// General tests that the header search paths detected by the driver and passed
// to CC1 are sane.
//
// Test a simulated installation of libc++ on Linux, both through sysroot and
// the installation path of Clang.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-SYSROOT %s
// CHECK-BASIC-LIBCXX-SYSROOT: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v1"
// CHECK-BASIC-LIBCXX-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxx_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxx_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXX-INSTALL %s
// CHECK-BASIC-LIBCXX-INSTALL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXX-INSTALL: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v1"
// CHECK-BASIC-LIBCXX-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxxv2_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXXV2-SYSROOT %s
// CHECK-BASIC-LIBCXXV2-SYSROOT: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXXV2-SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXXV2-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v2"
// CHECK-BASIC-LIBCXXV2-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_libcxxv2_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libcxxv2_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBCXXV2-INSTALL %s
// CHECK-BASIC-LIBCXXV2-INSTALL: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBCXXV2-INSTALL: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBCXXV2-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/bin/../include/c++/v2"
// CHECK-BASIC-LIBCXXV2-INSTALL: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
//
// Test Linux with both libc++ and libstdc++ installed.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -stdlib=libc++ \
// RUN:     -ccc-install-dir %S/Inputs/basic_linux_tree/usr/bin \
// RUN:     -resource-dir=%S/Inputs/resource_dir \
// RUN:     --sysroot=%S/Inputs/basic_linux_libstdcxx_libcxxv2_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-BASIC-LIBSTDCXX-LIBCXXV2-SYSROOT %s
// CHECK-BASIC-LIBSTDCXX-LIBCXXV2-SYSROOT: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-BASIC-LIBSTDCXX-LIBCXXV2-SYSROOT: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-BASIC-LIBSTDCXX-LIBCXXV2-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/include/c++/v2"
// CHECK-BASIC-LIBSTDCXX-LIBCXXV2-SYSROOT: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
//
// Test a very broken version of multiarch that shipped in Ubuntu 11.04.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i386-unknown-linux -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_11.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-11-04 %s
// CHECK-UBUNTU-11-04: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-UBUNTU-11-04: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-UBUNTU-11-04: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/i686-linux-gnu"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/i386-linux-gnu/gcc/i686-linux-gnu/4.5/../../../../../include/c++/4.5/backward"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-11-04: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-11-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_13.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-13-04 %s
// CHECK-UBUNTU-13-04: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-13-04: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-UBUNTU-13-04: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-13-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/c++/4.7"
// CHECK-UBUNTU-13-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/x86_64-linux-gnu/c++/4.7"
// CHECK-UBUNTU-13-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/c++/4.7/backward"
// CHECK-UBUNTU-13-04: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-13-04: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-UBUNTU-13-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/x86_64-linux-gnu"
// CHECK-UBUNTU-13-04: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-13-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnux32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04 %s
// CHECK-UBUNTU-14-04: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-14-04: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-UBUNTU-14-04: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8"
// CHECK-UBUNTU-14-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/x86_64-linux-gnu/c++/4.8/x32"
// CHECK-UBUNTU-14-04: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8/backward"
// CHECK-UBUNTU-14-04: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-14-04: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-UBUNTU-14-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/x86_64-linux-gnu"
// CHECK-UBUNTU-14-04: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-14-04: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
///
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm-linux-gnueabihf -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_13.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-13-04-CROSS %s
// CHECK-UBUNTU-13-04-CROSS: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-13-04-CROSS: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-UBUNTU-13-04-CROSS: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-13-04-CROSS: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.7/../../../../include/c++/4.7"
// CHECK-UBUNTU-13-04-CROSS: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.7/../../../../include/arm-linux-gnueabihf/c++/4.7"
// CHECK-UBUNTU-13-04-CROSS: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc-cross/arm-linux-gnueabihf/4.7/../../../../include/c++/4.7/backward"
// CHECK-UBUNTU-13-04-CROSS: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-UBUNTU-13-04-CROSS: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-UBUNTU-13-04-CROSS: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-13-04-CROSS: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Test Ubuntu/Debian's new version of multiarch, with -m32.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -m32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_13.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-13-04-M32 %s
// CHECK-UBUNTU-13-04-M32: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-13-04-M32: "-triple" "i386-unknown-linux-gnu"
// CHECK-UBUNTU-13-04-M32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-13-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/c++/4.7"
// CHECK-UBUNTU-13-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/x86_64-linux-gnu/c++/4.7/32"
// CHECK-UBUNTU-13-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.7/../../../../include/c++/4.7/backward"
//
// Test Ubuntu/Debian's Ubuntu 14.04 config variant, with -m32
// and an empty 4.9 directory.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -m32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-M32 %s
// CHECK-UBUNTU-14-04-M32: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-14-04-M32: "-triple" "i386-unknown-linux-gnu"
// CHECK-UBUNTU-14-04-M32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8"
// CHECK-UBUNTU-14-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/x86_64-linux-gnu/c++/4.8/32"
// CHECK-UBUNTU-14-04-M32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.8/../../../../include/c++/4.8/backward"
//
// Test Ubuntu/Debian's Ubuntu 14.04 with -m32 and an i686 cross compiler
// installed rather than relying on multilib. Also happens to look like an
// actual i686 Ubuntu system.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -m32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree2 \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-I686 %s
// CHECK-UBUNTU-14-04-I686: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-14-04-I686: "-triple" "i386-unknown-linux-gnu"
// CHECK-UBUNTU-14-04-I686: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-I686: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.8/../../../../include/c++/4.8"
// CHECK-UBUNTU-14-04-I686: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.8/../../../../include/i386-linux-gnu/c++/4.8"
// CHECK-UBUNTU-14-04-I686: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.8/../../../../include/c++/4.8/backward"
//
// Test Ubuntu/Debian's Ubuntu 14.04 for powerpc64le
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target powerpc64le-unknown-linux-gnu -m32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/ubuntu_14.04_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-UBUNTU-14-04-PPC64LE %s
// CHECK-UBUNTU-14-04-PPC64LE: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-UBUNTU-14-04-PPC64LE: "-triple" "powerpc64le-unknown-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../../include/c++/4.8"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../../include/powerpc64le-linux-gnu/c++/4.8"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64le-linux-gnu/4.8/../../../../include/c++/4.8/backward"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/powerpc64le-linux-gnu"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-UBUNTU-14-04-PPC64LE: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Thoroughly exercise the Debian multiarch environment.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i686-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86 %s
// CHECK-DEBIAN-X86: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-X86: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-X86: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5/i686-linux-gnu"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-X86: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/i386-linux-gnu"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-X86: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-X86-64 %s
// CHECK-DEBIAN-X86-64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-X86-64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-X86-64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5/x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-X86-64: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/x86_64-linux-gnu"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-X86-64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target powerpc-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC %s
// CHECK-DEBIAN-PPC: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-PPC: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-PPC: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5/powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-PPC: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/powerpc-linux-gnu"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-PPC: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target powerpc64-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_multiarch_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-PPC64 %s
// CHECK-DEBIAN-PPC64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-PPC64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-PPC64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5/powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/powerpc64-linux-gnu/4.5/../../../../include/c++/4.5/backward"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-PPC64: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/powerpc64-linux-gnu"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-PPC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Test Gentoo's weirdness both before and after they changed it in their GCC
// 4.6.4 release.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.6.2_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-6-2 %s
// CHECK-GENTOO-4-6-2: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-6-2: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-6-2: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-6-2: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/include/g++-v4"
// CHECK-GENTOO-4-6-2: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/include/g++-v4/x86_64-pc-linux-gnu"
// CHECK-GENTOO-4-6-2: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/include/g++-v4/backward"
// CHECK-GENTOO-4-6-2: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-6-2: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-6-2: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-6-2: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.6.4_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-6-4 %s
// CHECK-GENTOO-4-6-4: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-6-4: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-6-4: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-6-4: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.4/include/g++-v4.6"
// CHECK-GENTOO-4-6-4: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.4/include/g++-v4.6/x86_64-pc-linux-gnu"
// CHECK-GENTOO-4-6-4: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.6.4/include/g++-v4.6/backward"
// CHECK-GENTOO-4-6-4: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-6-4: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-6-4: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-6-4: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.9.3_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-3 %s
// CHECK-GENTOO-4-9-3: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-3: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-3: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-3: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-3: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/x86_64-pc-linux-gnu"
// CHECK-GENTOO-4-9-3: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-3: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-3: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-3: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-3: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Test support for Gentoo's gcc-config -- clang should prefer the older
// (4.9.3) version over the newer (5.4.0) due to preference specified
// in /etc/env.d/gcc/x86_64-pc-linux-gnu.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-3 %s
//
// Test that gcc-config support does not break multilib.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnux32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-3-X32 %s
// CHECK-GENTOO-4-9-3-X32: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-3-X32: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-3-X32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-3-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-3-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/x86_64-pc-linux-gnu/x32"
// CHECK-GENTOO-4-9-3-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-3-X32: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-3-X32: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-3-X32: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-3-X32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i386-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_multi_version_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-3-32 %s
// CHECK-GENTOO-4-9-3-32: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-3-32: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-3-32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-3-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-3-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/x86_64-pc-linux-gnu/32"
// CHECK-GENTOO-4-9-3-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.3/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-3-32: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-3-32: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-3-32: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-3-32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Test support for parsing Gentoo's gcc-config -- clang should parse the
// /etc/env.d/gcc/config-x86_64-pc-linux-gnu file to find CURRENT gcc used.
// Then should pick the multilibs from version 4.9.x specified in
// /etc/env.d/gcc/x86_64-pc-linux-gnu-4.9.3.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.9.x_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-X %s
//
// CHECK-GENTOO-4-9-X: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-X: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-X: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-X: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-X: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/x86_64-pc-linux-gnu"
// CHECK-GENTOO-4-9-X: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-X: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-X: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-X: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-X: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnux32 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.9.x_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-X-X32 %s
// CHECK-GENTOO-4-9-X-X32: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-X-X32: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-X-X32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-X-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-X-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/x86_64-pc-linux-gnu/x32"
// CHECK-GENTOO-4-9-X-X32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-X-X32: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-X-X32: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-X-X32: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-X-X32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target i386-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/gentoo_linux_gcc_4.9.x_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-GENTOO-4-9-X-32 %s
// CHECK-GENTOO-4-9-X-32: "{{.*}}clang{{.*}}" "-cc1"
// CHECK-GENTOO-4-9-X-32: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-GENTOO-4-9-X-32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-GENTOO-4-9-X-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3"
// CHECK-GENTOO-4-9-X-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/x86_64-pc-linux-gnu/32"
// CHECK-GENTOO-4-9-X-32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-linux-gnu/4.9.x/include/g++-v4.9.3/backward"
// CHECK-GENTOO-4-9-X-32: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-GENTOO-4-9-X-32: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-GENTOO-4-9-X-32: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-GENTOO-4-9-X-32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Check header search on Debian 6 / MIPS64
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target mips64-unknown-linux-gnuabi64 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64-GNUABI %s
// CHECK-MIPS64-GNUABI: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-MIPS64-GNUABI: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-MIPS64-GNUABI: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-MIPS64-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../../include/c++/4.9"
// CHECK-MIPS64-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../../include/c++/4.9/mips64-linux-gnuabi64"
// CHECK-MIPS64-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64-linux-gnuabi64/4.9/../../../../include/c++/4.9/backward"
// CHECK-MIPS64-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-MIPS64-GNUABI: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-MIPS64-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/mips64-linux-gnuabi64"
// CHECK-MIPS64-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-MIPS64-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
//
// Check header search on Debian 6 / MIPS64
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target mips64el-unknown-linux-gnuabi64 -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_6_mips64_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-MIPS64EL-GNUABI %s
// CHECK-MIPS64EL-GNUABI: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-MIPS64EL-GNUABI: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-MIPS64EL-GNUABI: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-MIPS64EL-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../include/c++/4.9"
// CHECK-MIPS64EL-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../include/c++/4.9/mips64el-linux-gnuabi64"
// CHECK-MIPS64EL-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/mips64el-linux-gnuabi64/4.9/../../../../include/c++/4.9/backward"
// CHECK-MIPS64EL-GNUABI: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-MIPS64EL-GNUABI: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-MIPS64EL-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/mips64el-linux-gnuabi64"
// CHECK-MIPS64EL-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-MIPS64EL-GNUABI: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Check header search on Debian 8 / Sparc
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target sparc-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc_multilib_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC32 %s
// CHECK-DEBIAN-SPARC32: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-SPARC32: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-SPARC32: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/c++/4.9"
// CHECK-DEBIAN-SPARC32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/sparc-linux-gnu/c++/4.9"
// CHECK-DEBIAN-SPARC32: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/c++/4.9/backward"
// CHECK-DEBIAN-SPARC32: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-SPARC32: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-SPARC32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/sparc-linux-gnu"
// CHECK-DEBIAN-SPARC32: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-SPARC32: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Check header search on Debian 8 / Sparc, with the oldstyle multilib packages
// RUN: %clang -no-canonical-prefixes -m64 %s -### -fsyntax-only 2>&1 \
// RUN:     -target sparc-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc_multilib_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC32-LIB64 %s
// CHECK-DEBIAN-SPARC32-LIB64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-SPARC32-LIB64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-SPARC32-LIB64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/c++/4.9"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/sparc-linux-gnu/c++/4.9/64"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc-linux-gnu/4.9/../../../../include/c++/4.9/backward"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
/* TODO: GCC 4.9 includes the following dir in its search path, which
   seems questionable. Clang doesn't. Not sure if clang should be
   doing that too. */
// CHECK-DEBIAN-SPARC32-LIB64-todo: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/sparc-linux-gnu"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-SPARC32-LIB64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Check header search on Debian 8 / Sparc64
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target sparc64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/debian_8_sparc64_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-DEBIAN-SPARC64 %s
// CHECK-DEBIAN-SPARC64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-DEBIAN-SPARC64: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// CHECK-DEBIAN-SPARC64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-DEBIAN-SPARC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../../include/c++/4.9"
// CHECK-DEBIAN-SPARC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../../include/sparc64-linux-gnu/c++/4.9"
// CHECK-DEBIAN-SPARC64: "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/sparc64-linux-gnu/4.9/../../../../include/c++/4.9/backward"
// CHECK-DEBIAN-SPARC64: "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-DEBIAN-SPARC64: "-internal-isystem" "[[RESOURCE_DIR]]{{/|\\\\}}include"
// CHECK-DEBIAN-SPARC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include/sparc64-linux-gnu"
// CHECK-DEBIAN-SPARC64: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-DEBIAN-SPARC64: "-internal-externc-isystem" "[[SYSROOT]]/usr/include"

// Check header search on OpenEmbedded ARM.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm-oe-linux-gnueabi -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/openembedded_arm_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-ARM %s

// CHECK-OE-ARM: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-OE-ARM: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-OE-ARM: "-internal-isystem" "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../include/c++/6.3.0"
// CHECK-OE-ARM: "-internal-isystem" "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../include/c++/6.3.0/backward"

// Check header search on OpenEmbedded AArch64.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target aarch64-oe-linux -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/openembedded_aarch64_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-OE-AARCH64 %s

// CHECK-OE-AARCH64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-OE-AARCH64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-OE-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../include/c++/6.3.0"
// CHECK-OE-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../include/c++/6.3.0/backward"
