// Test the driver's control over the PIC behavior. These consist of tests of
// the relocation model flags and the pic level flags passed to CC1.
//
// CHECK-NO-PIC: "-mrelocation-model" "static"
// CHECK-NO-PIC-NOT: "-pic-level"
// CHECK-NO-PIC-NOT: "-pic-is-pie"
//
// CHECK-PIC1: "-mrelocation-model" "pic"
// CHECK-PIC1: "-pic-level" "1"
// CHECK-PIC1-NOT: "-pic-is-pie"
//
// CHECK-PIC2: "-mrelocation-model" "pic"
// CHECK-PIC2: "-pic-level" "2"
// CHECK-PIC2-NOT: "-pic-is-pie"
//
// CHECK-STATIC: "-static"
// CHECK-NO-STATIC-NOT: "-static"
//
// CHECK-PIE1: "-mrelocation-model" "pic"
// CHECK-PIE1: "-pic-level" "1"
// CHECK-PIE1: "-pic-is-pie"
//
// CHECK-PIE2: "-mrelocation-model" "pic"
// CHECK-PIE2: "-pic-level" "2"
// CHECK-PIE2: "-pic-is-pie"
//
// CHECK-PIE-LD: "{{.*}}ld{{(.exe)?}}"
// CHECK-PIE-LD: "-pie"
// CHECK-PIE-LD: "Scrt1.o" "crti.o" "crtbeginS.o"
// CHECK-PIE-LD: "crtendS.o" "crtn.o"
//
// CHECK-NOPIE-LD: "-nopie"
//
// CHECK-DYNAMIC-NO-PIC-32: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC-32-NOT: "-pic-level"
// CHECK-DYNAMIC-NO-PIC-32-NOT: "-pic-is-pie"
//
// CHECK-DYNAMIC-NO-PIC-64: "-mrelocation-model" "dynamic-no-pic"
// CHECK-DYNAMIC-NO-PIC-64: "-pic-level" "2"
// CHECK-DYNAMIC-NO-PIC-64-NOT: "-pic-is-pie"
//
// CHECK-NON-DARWIN-DYNAMIC-NO-PIC: error: unsupported option '-mdynamic-no-pic' for target 'i386-unknown-unknown'
//
// CHECK-NO-PIE-NOT: "-pie"
//
// CHECK-NO-UNUSED-ARG-NOT: argument unused during compilation
//
// RUN: %clang -c %s -target i386-unknown-unknown -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Check that PIC and PIE flags obey last-match-wins. If the last flag is
// a no-* variant, regardless of which variant or which flags precede it, we
// get no PIC.
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// Last-match-wins where both pic and pie are specified.
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIE -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Last-match-wins when selecting level 1 vs. level 2.
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-unknown-unknown -fPIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC1
// RUN: %clang -c %s -target i386-unknown-unknown -fpic -fPIE -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-unknown-unknown -fpie -fPIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Make sure -pie is passed to along to ld and that the right *crt* files
// are linked in.
// RUN: %clang %s -target i386-unknown-freebsd -fPIE -pie -### \
// RUN: --gcc-toolchain="" -rtlib=platform \
// RUN: --sysroot=%S/Inputs/basic_freebsd_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE-LD
// RUN: %clang %s -target i386-linux-gnu -fPIE -pie -### \
// RUN: --gcc-toolchain="" -rtlib=platform \
// RUN: --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE-LD
// RUN: %clang %s -target i386-linux-gnu -fPIC -pie -### \
// RUN: --gcc-toolchain="" -rtlib=platform \
// RUN: --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE-LD
//
// Disregard any of the PIC-specific flags if we have a trump-card flag.
// RUN: %clang -c %s -target i386-unknown-unknown -mkernel -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC

// The -static argument *doesn't* override PIC: -static only affects
// linking, and -fPIC only affects code generation.
// RUN: %clang -c %s -target i386-unknown-unknown -static -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang %s -target i386-linux-gnu -static -fPIC -### \
// RUN: --gcc-toolchain="" \
// RUN: --sysroot=%S/Inputs/basic_linux_tree 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-STATIC
//
// On Linux, disregard -pie if we have -shared.
// RUN: %clang %s -target i386-unknown-linux -shared -pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIE
//
// On Musl Linux, PIE is enabled by default, but can be disabled.
// RUN: %clang -c %s -target x86_64-linux-musl -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i686-linux-musl -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target armv6-linux-musleabihf -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target armv7-linux-musleabihf -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang %s -target x86_64-linux-musl -nopie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIE
// RUN: %clang %s -target x86_64-linux-musl -pie -nopie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIE
// RUN: %clang %s -target x86_64-linux-musl -nopie -pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Darwin is a beautiful and unique snowflake when it comes to these flags.
// When targeting a 32-bit darwin system, only level 2 is supported. On 64-bit
// targets, there is simply nothing you can do, there is no PIE, there is only
// PIC when it comes to compilation.
// RUN: %clang -c %s -target i386-apple-darwin -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-apple-darwin -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIC -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target i386-apple-darwin -fno-PIC -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target x86_64-apple-darwin -fno-PIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-apple-darwin -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-apple-darwin -fpic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-apple-darwin -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-apple-darwin -fPIC -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-UNUSED-ARG
//
// Darwin gets even more special with '-mdynamic-no-pic'. This flag is only
// valid on Darwin, and it's behavior is very strange but needs to remain
// consistent for compatibility.
// RUN: %clang -c %s -target i386-unknown-unknown -mdynamic-no-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NON-DARWIN-DYNAMIC-NO-PIC
// RUN: %clang -c %s -target i386-apple-darwin -mdynamic-no-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-32
// RUN: %clang -c %s -target i386-apple-darwin -mdynamic-no-pic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-32
// RUN: %clang -c %s -target i386-apple-darwin -mdynamic-no-pic -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-32
// RUN: %clang -c %s -target x86_64-apple-darwin -mdynamic-no-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-64
// RUN: %clang -c %s -target x86_64-apple-darwin -mdynamic-no-pic -fno-pic -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-64
// RUN: %clang -c %s -target x86_64-apple-darwin -mdynamic-no-pic -fpie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DYNAMIC-NO-PIC-64
//
// Checks for ARM+Apple+IOS including -fapple-kext, -mkernel, and iphoneos
// version boundaries.
// RUN: %clang -c %s -target armv7-apple-ios6 -fapple-kext -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target armv7-apple-ios6 -mkernel -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target arm64-apple-ios7 -mkernel -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -x assembler -c %s -target arm64-apple-ios7 -mkernel -no-integrated-as -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-STATIC
// RUN: %clang -c %s -target armv7k-apple-watchos1 -fapple-kext -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-apple-driverkit -fapple-kext -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target armv7-apple-ios5 -fapple-kext -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target armv7-apple-ios6 -fapple-kext -static -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target armv7-apple-unknown-macho -static -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// On OpenBSD, PIE is enabled by default, but can be disabled.
// RUN: %clang -c %s -target amd64-pc-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target i386-pc-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target aarch64-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target arm-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target mips64-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target mips64el-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
// RUN: %clang -c %s -target powerpc-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target sparc64-unknown-openbsd -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target i386-pc-openbsd -fno-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// On OpenBSD, -nopie needs to be passed through to the linker.
// RUN: %clang %s -target i386-pc-openbsd -nopie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIE-LD
// Try with the alias
// RUN: %clang %s -target i386-pc-openbsd -no-pie -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOPIE-LD
//
// On Android PIC is enabled by default, and PIE is enabled by default starting
// with API16.
// RUN: %clang -c %s -target i686-linux-android24 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// RUN: %clang -c %s -target arm-linux-androideabi24 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// RUN: %clang -c %s -target mipsel-linux-android24 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE1
//
// 64-bit Android targets are always PIE.
// RUN: %clang -c %s -target aarch64-linux-android -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target aarch64-linux-android24 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target arm64-linux-android -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
//
// Default value of PIE can be overwritten, even on 64-bit targets.
// RUN: %clang -c %s -target arm-linux-androideabi -fPIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIE2
// RUN: %clang -c %s -target aarch64-linux-android -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -c %s -target aarch64-linux-android24 -fno-PIE -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
//
// On Windows x86_64 and aarch64 PIC is enabled by default
// RUN: %clang -c %s -target x86_64-pc-windows-msvc18.0.0 -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target x86_64-pc-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target aarch64-windows-msvc -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -c %s -target aarch64-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
//
// On MinGW, allow specifying -fPIC & friends but ignore them
// RUN: %clang -fno-PIC -c %s -target x86_64-pc-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -fPIC -c %s -target i686-pc-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
// RUN: %clang -fno-PIC -c %s -target aarch64-pc-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-PIC2
// RUN: %clang -fPIC -c %s -target armv7-pc-windows-gnu -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-NO-PIC
