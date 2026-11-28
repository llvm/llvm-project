// RUN: %clang_cc1 -E -dM -triple=riscv32-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=riscv64-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=arm-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=mips-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=powerpc-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=sparc-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s
// RUN: %clang_cc1 -E -dM -triple=i686-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefix=RTEMS %s

// RUN: %clang_cc1 -x c++ -E -dM -triple=riscv32-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=RTEMS,RTEMS-CXX %s
// RUN: %clang_cc1 -x c++ -E -dM -triple=riscv64-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=RTEMS,RTEMS-CXX %s
// RUN: %clang_cc1 -x c++ -E -dM -triple=sparc-unknown-rtems < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=RTEMS,RTEMS-CXX %s

// RTEMS-DAG: #define __rtems__ 1
// RTEMS-CXX-DAG: #define _GNU_SOURCE 1
