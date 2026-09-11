// The following verifies that __int128 is not supported
// RUN: %clang_cc1 -triple s390x-unknown-zos -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple s390x-unknown-zos -fsyntax-only -target-cpu arch10 -verify %s
// RUN: not %clang -msoft-float --target=s390x-unknown-zos %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-1 %s < %t.err
// RUN: not %clang -mvx -march=zEC12 --target=s390x-unknown-linux-gnu %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-2 %s < %t.err
// RUN: not %clang -mvx -march=zEC12 -mhard-float --target=s390x-unknown-linux-gnu %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-2 %s < %t.err
// RUN: not %clang -mvx -march=zEC12 -msoft-float --target=s390x-unknown-linux-gnu %s 2> %t.err
// RUN: not FileCheck -check-prefix=CHECK-2 %s < %t.err

__int128 pass_int128(__int128 arg) { return arg; } // expected-error {{__int128 is not supported on this target}} expected-error {{__int128 is not supported on this target}}

// CHECK-1: error: unsupported option '-msoft-float' for target 's390x-unknown-zos'
// CHECK-2: error: '-mvx' option requires at least '-march=arch11', current 'zEC12' is too low
