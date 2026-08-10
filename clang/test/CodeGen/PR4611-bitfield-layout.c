// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s
//
// CHECK: struct.object_entry = type { i32 }

struct object_entry {
       unsigned int type:3, pack_id:16, depth:13;
} entries;
