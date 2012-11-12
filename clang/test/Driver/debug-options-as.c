// cygming have not supported integrated-as yet.
// XFAIL: cygwin,mingw32
//
// Check to make sure clang is somewhat picky about -g options.
// (Delived from debug-options.c)
// rdar://10383444
// RUN: %clang -### -c -save-temps -integrated-as -g %s 2>&1 \
// RUN:   | FileCheck -check-prefix=SAVE %s
//
// SAVE: "-cc1as"
// SAVE-NOT: "-g"

// Check to make sure clang with -g on a .s file gets passed.
// rdar://9275556
// RUN: touch %t.s
// RUN: %clang -### -c -integrated-as -g %t.s 2>&1 \
// RUN:   | FileCheck -check-prefix=S %s
//
// S: "-cc1as"
// S: "-g"
