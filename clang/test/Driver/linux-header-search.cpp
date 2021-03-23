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


// Check header search on OpenEmbedded ARM.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target arm-oe-linux-gnueabi -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/openembedded_arm_linux_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-OE-ARM %s

// CHECK-OE-ARM: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-OE-ARM: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-OE-ARM: "-internal-isystem" "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../include/c++/6.3.0"
// CHECK-OE-ARM: "-internal-isystem" "[[SYSROOT]]/usr/lib/arm-oe-linux-gnueabi/6.3.0/../../../include/c++/6.3.0/backward"

// Check header search on OpenEmbedded AArch64.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target aarch64-oe-linux -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/openembedded_aarch64_linux_tree \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=CHECK-OE-AARCH64 %s

// CHECK-OE-AARCH64: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-OE-AARCH64: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-OE-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../include/c++/6.3.0"
// CHECK-OE-AARCH64: "-internal-isystem" "[[SYSROOT]]/usr/lib64/aarch64-oe-linux/6.3.0/../../../include/c++/6.3.0/backward"

// Check header search with Cray's gcc package.
// RUN: %clang -no-canonical-prefixes %s -### -fsyntax-only 2>&1 \
// RUN:     -target x86_64-unknown-linux-gnu -stdlib=libstdc++ \
// RUN:     --sysroot=%S/Inputs/cray_suse_gcc_tree \
// RUN:     --gcc-toolchain="%S/Inputs/cray_suse_gcc_tree/opt/gcc/8.2.0/snos" \
// RUN:   | FileCheck --check-prefix=CHECK-CRAY-X86 %s

// CHECK-CRAY-X86: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK-CRAY-X86: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-CRAY-X86: "-internal-isystem" "[[SYSROOT]]/opt/gcc/8.2.0/snos/lib/gcc/x86_64-suse-linux/8.2.0/../../../../include/g++"
// CHECK-CRAY-X86: "-internal-isystem" "[[SYSROOT]]/opt/gcc/8.2.0/snos/lib/gcc/x86_64-suse-linux/8.2.0/../../../../include/g++/backward"
