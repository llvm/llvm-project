// RUN: touch %t.o
//
// RUN: %clang -target ppc64le-unknown-linux -### %t.o -flto 2>&1 \
// RUN:     -Wl,-plugin-opt=foo -O3 \
// RUN:     -fexperimental-new-pass-manager \
// RUN:     | FileCheck %s
// CHECK: "-plugin-opt=new-pass-manager"
