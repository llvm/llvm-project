// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o %t
// RUN: grep "struct.object_entry = type { i8, \[2 x i8\], i8 }" %t

struct object_entry {
       unsigned int type:3, pack_id:16, depth:13;
} entries;
