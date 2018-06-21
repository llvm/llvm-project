// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -fms-compatibility -o - < %s | FileCheck %s

char __declspec(align(8192)) x;
// CHECK-DAG: @x = global i8 0, align 8192

typedef char __declspec(align(8192)) T;
T y;
// CHECK-DAG: @y = global i8 0, align 8192

T __declspec(align(8192)) z;
// CHECK-DAG: @z = global i8 0, align 8192

int __declspec(align(16)) redef;
int __declspec(align(32)) redef = 8;
// CHECK-DAG: @redef = global i32 8, align 32

struct __declspec(align(64)) S {
  char fd;
} s;
// CHECK-DAG: @s = global %struct.S zeroinitializer, align 64

struct Wrap {
  struct S x;
} w;
// CHECK-DAG: @w = global %struct.Wrap zeroinitializer, align 64
