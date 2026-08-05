// RUN: %clang_cc1 -E -dM -triple=aarch64-unknown-fuchsia < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=FUCHSIA %s
// RUN: %clang_cc1 -E -dM -triple=arm-unknown-fuchsia < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=FUCHSIA %s
// RUN: %clang_cc1 -E -dM -triple=riscv64-unknown-fuchsia < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=FUCHSIA %s
// RUN: %clang_cc1 -E -dM -triple=x86_64-unknown-fuchsia < /dev/null \
// RUN:   | FileCheck -match-full-lines -check-prefixes=FUCHSIA %s

// FUCHSIA-DAG: #define __Fuchsia__ 1
// FUCHSIA-DAG: #define __Fuchsia_API_level__ {{[0-9]+}}
