// Check that -iwithprefix falls into the "after" search list.
//
// RUN: rm -rf %t.tmps
// RUN: mkdir -p %t.tmps/first %t.tmps/second
// RUN: %clang_cc1 -triple x86_64-unknown-unknown \
// RUN:   -iprefix %t.tmps/ -iwithprefix second \
// RUN:    -isystem %t.tmps/first -v %s 2> %t.out
// RUN: FileCheck %s < %t.out

// CHECK: #include <...> search starts here:
// CHECK: {{.*}}.tmps/first
// CHECK: {{/|\\}}clang{{/|\\}}{{[.0-9]+}}{{/|\\}}include
// CHECK: {{.*}}.tmps/second
// CHECK-NOT: {{.*}}.tmps
