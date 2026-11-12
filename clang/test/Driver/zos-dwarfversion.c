// RUN: %clang -target s390x-none-zos -g -S -emit-llvm %s -o - | FileCheck %s

// CHECK: !"Dwarf Version", i32 4
