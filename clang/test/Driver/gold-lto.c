// RUN: touch %t.o
// RUN: %clang -target x86_64-pc-linux-gnu -### %t.o -O4 -Wl,-plugin-opt=foo 2> %t.log
// RUN: FileCheck %s < %t.log

// CHECK: "-plugin" "{{.*}}/LLVMgold.so"
// CHECK: "-plugin-opt=foo"
