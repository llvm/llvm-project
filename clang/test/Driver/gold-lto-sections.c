// RUN: touch %t.o
//
// RUN: %clang -target x86_64-unknown-linux -### %t.o -flto 2>&1 \
// RUN:     -Wl,-plugin-opt=foo -O3 \
// RUN:     -ffunction-sections -fdata-sections \
// RUN:     | FileCheck %s
// CHECK: "-plugin-opt=-function-sections"
// CHECK: "-plugin-opt=-data-sections"
