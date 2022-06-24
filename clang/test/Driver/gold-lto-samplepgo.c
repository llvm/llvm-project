// RUN: touch %t.o
//
// RUN: %clang -target x86_64-unknown-linux -### %t.o -flto 2>&1 \
// RUN:     -Wl,-plugin-opt=foo -O3 \
// RUN:     -fprofile-sample-use=%s \
// RUN:     | FileCheck %s
// CHECK: -plugin-opt=sample-profile=
