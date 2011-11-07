// General tests that ld invocations on Linux targets sane. Note that we use
// sysroot to make these tests independent of the host system.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0/crtbegin.o"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-LD-32: "-L[[SYSROOT]]/lib"
// CHECK-LD-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple x86_64-unknown-linux \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-LD-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/crtbegin.o"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-LD-64: "-L[[SYSROOT]]/lib"
// CHECK-LD-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-32 %s
// CHECK-32-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-32: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0/crtbegin.o"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-32-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m64 \
// RUN:     --sysroot=%S/Inputs/multilib_32bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-32-TO-64 %s
// CHECK-32-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-32-TO-64: "{{.*}}/usr/lib/gcc/i386-unknown-linux/4.6.0/64/crtbegin.o"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../../../i386-unknown-linux/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/i386-unknown-linux/4.6.0/../../.."
// CHECK-32-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-32-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple x86_64-unknown-linux -m64 \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-64 %s
// CHECK-64-TO-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/crtbegin.o"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/../lib64"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-64-TO-64: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-64: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple x86_64-unknown-linux -m32 \
// RUN:     --sysroot=%S/Inputs/multilib_64bit_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-64-TO-32 %s
// CHECK-64-TO-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-64-TO-32: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32/crtbegin.o"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/../lib32"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../../../x86_64-unknown-linux/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0/../../.."
// CHECK-64-TO-32: "-L[[SYSROOT]]/lib"
// CHECK-64-TO-32: "-L[[SYSROOT]]/usr/lib"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/fake_install_tree/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INSTALL-DIR-32 %s
// CHECK-INSTALL-DIR-32: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-INSTALL-DIR-32: "{{.*}}/Inputs/fake_install_tree/bin/../lib/gcc/i386-unknown-linux/4.7.0/crtbegin.o"
// CHECK-INSTALL-DIR-32: "-L{{.*}}/Inputs/fake_install_tree/bin/../lib/gcc/i386-unknown-linux/4.7.0"
//
// Check that with 64-bit builds, we don't actually use the install directory
// as its version of GCC is lower than our sysrooted version.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple x86_64-unknown-linux -m64 \
// RUN:     -ccc-install-dir %S/Inputs/fake_install_tree/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-INSTALL-DIR-64 %s
// CHECK-INSTALL-DIR-64: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-INSTALL-DIR-64: "{{.*}}/usr/lib/gcc/x86_64-unknown-linux/4.6.0/crtbegin.o"
// CHECK-INSTALL-DIR-64: "-L[[SYSROOT]]/usr/lib/gcc/x86_64-unknown-linux/4.6.0"
//
// Check that we support unusual patch version formats, including missing that
// component.
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing1/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION1 %s
// CHECK-GCC-VERSION1: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION1: "{{.*}}/Inputs/gcc_version_parsing1/bin/../lib/gcc/i386-unknown-linux/4.7/crtbegin.o"
// CHECK-GCC-VERSION1: "-L{{.*}}/Inputs/gcc_version_parsing1/bin/../lib/gcc/i386-unknown-linux/4.7"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing2/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION2 %s
// CHECK-GCC-VERSION2: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION2: "{{.*}}/Inputs/gcc_version_parsing2/bin/../lib/gcc/i386-unknown-linux/4.7.x/crtbegin.o"
// CHECK-GCC-VERSION2: "-L{{.*}}/Inputs/gcc_version_parsing2/bin/../lib/gcc/i386-unknown-linux/4.7.x"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing3/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION3 %s
// CHECK-GCC-VERSION3: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION3: "{{.*}}/Inputs/gcc_version_parsing3/bin/../lib/gcc/i386-unknown-linux/4.7.99-rc5/crtbegin.o"
// CHECK-GCC-VERSION3: "-L{{.*}}/Inputs/gcc_version_parsing3/bin/../lib/gcc/i386-unknown-linux/4.7.99-rc5"
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -ccc-host-triple i386-unknown-linux -m32 \
// RUN:     -ccc-install-dir %S/Inputs/gcc_version_parsing4/bin \
// RUN:     --sysroot=%S/Inputs/basic_linux_tree \
// RUN:   | FileCheck --check-prefix=CHECK-GCC-VERSION4 %s
// CHECK-GCC-VERSION4: "{{.*}}ld{{(.exe)?}}" "--sysroot=[[SYSROOT:[^"]+]]"
// CHECK-GCC-VERSION4: "{{.*}}/Inputs/gcc_version_parsing4/bin/../lib/gcc/i386-unknown-linux/4.7.99/crtbegin.o"
// CHECK-GCC-VERSION4: "-L{{.*}}/Inputs/gcc_version_parsing4/bin/../lib/gcc/i386-unknown-linux/4.7.99"
