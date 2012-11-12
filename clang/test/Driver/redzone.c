// RUN: %clang -target i386-unknown-unknown -mno-red-zone %s -S -emit-llvm -o %t.log
// RUN: grep 'noredzone' %t.log
// RUN: %clang -target i386-unknown-unknown -mred-zone %s -S -emit-llvm -o %t.log
// RUN: grep -v 'noredzone' %t.log 

int foo() { return 42; }
