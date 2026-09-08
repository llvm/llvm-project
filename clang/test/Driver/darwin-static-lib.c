// -D and -no_warning_for_no_symbols are on by default.
// RUN: %clang -target i386-apple-darwin9 %s -### --emit-static-lib 2>&1 | FileCheck %s
// CHECK: "{{.*}}libtool" "-static" "-D" "-no_warning_for_no_symbols" "-o" "a.out" "{{.*o}}"

// RUN: %clang -target i386-apple-darwin9 %s -### --emit-static-lib -o libfoo.a 2>&1 | FileCheck %s --check-prefix=OUTPUT
// OUTPUT: "{{.*}}libtool" "-static" "-D" "-no_warning_for_no_symbols" "-o" "libfoo.a" "{{.*o}}"

// RUN: touch %t1.o %t2.o

// Adding the explicit flags for -D and -no_warning_for_no_symbols doesn't
// double up the arguments passed to libtool.
// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --static-lib-deterministic --no-static-lib-warn-no-symbols 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// DEFAULT: "{{.*}}libtool" "-static" "-D" "-no_warning_for_no_symbols" "-o" "a.out" "{{.*}}1.o" "{{.*}}2.o"

// Last one wins with contradictory arguments.
// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --no-static-lib-deterministic --static-lib-deterministic 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DEFAULT

// -D and -no_warning_for_no_symbols can be turned off.
// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --no-static-lib-deterministic --static-lib-warn-no-symbols 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NEITHER
// NEITHER:     "-static"
// NEITHER-NOT: "-D"
// NEITHER-NOT: "-no_warning_for_no_symbols"
// NEITHER:     "-o"

// -arch_only is derived from the target, and only when asked for. The Mach-O
// arch name is used, not the clang one. i.e. the clang driver turns
// armv7k-apple-watchos8.0 into thumbv7k-apple-watchos8.0.0, but -arch_only uses
// the MachO name armv7k. -force_cpusubtype_ALL isn't used by libtool.
// RUN: %clang -target armv7k-apple-watchos8.0 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --static-lib-target-arch-only 2>&1 | FileCheck %s --check-prefix=ARMV7K
// ARMV7K: "-static" "-arch_only" "armv7k" "-D"
// ARMV7K-NOT: "-force_cpusubtype_ALL"

// Firmware uses the full triple.
// RUN: %clang -target armv7em-apple-firmware1.0 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --static-lib-target-arch-only 2>&1 | FileCheck %s --check-prefix=FIRMWARE
// FIRMWARE: "-arch_only" "thumbv7em-apple-firmware1.0.0"

// -arch_only on the clang command line is ignored and not passed on.
// RUN: %clang -target armv7k-apple-watchos8.0 -### --emit-static-lib %t1.o %t2.o \
// RUN:     --static-lib-target-arch-only -arch_only arm32_64 2>&1 | \
// RUN:     FileCheck %s --check-prefix=ARCH-ONLY
// ARCH-ONLY: warning: argument unused during compilation: '-arch_only arm32_64'
// ARCH-ONLY: "-static" "-arch_only"
// ARCH-ONLY-NOT: "arm32_64"
// ARCH-ONLY: "armv7k" "-D"

// sysroot becomes -syslibroot, with --sysroot= taking priority over -isysroot.
// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     -isysroot %S/Inputs/MacOSX15.1.sdk --sysroot=/tmp/sysroot 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SYSROOT
// SYSROOT: "-syslibroot" "/tmp/sysroot"

// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     -isysroot %S/Inputs/MacOSX15.1.sdk 2>&1 | FileCheck %s --check-prefix=ISYSROOT
// ISYSROOT: "-syslibroot" "{{.*}}MacOSX15.1.sdk"

// -L, -filelist are forwarded as is. -filelist doesn't get doubled up as an input file.
// -Xstatic-lib-tool passes through.
// RUN: %clang -target i386-apple-darwin9 -### --emit-static-lib %t1.o %t2.o \
// RUN:     -L/tmp/first -L/tmp/second -Xstatic-lib-tool -dependency_info -Xstatic-lib-tool deps.dat \
// RUN:     -filelist objs.txt 2>&1 | FileCheck %s --check-prefix=PASSTHROUGH
// PASSTHROUGH-DAG: "-L/tmp/first" "-L/tmp/second"
// PASSTHROUGH-DAG: "-dependency_info" "deps.dat"
// PASSTHROUGH-DAG: "-filelist" "objs.txt"
// PASSTHROUGH-DAG: "{{.*}}1.o" "{{.*}}2.o"
// PASSTHROUGH-NOT: "-filelist"

// Multiple -arch produces one libtool job per arch plus a lipo.
// RUN: %clang -target x86_64-apple-macos14 -### --emit-static-lib %t1.o %t2.o \
// RUN:     -arch x86_64 -arch arm64 -o libfoo.a 2>&1 | FileCheck %s --check-prefix=UNIVERSAL
// UNIVERSAL: "{{.*}}libtool" "-static"
// UNIVERSAL: "{{.*}}libtool" "-static"
// UNIVERSAL: "{{.*}}lipo" "-create" "-output" "libfoo.a"
