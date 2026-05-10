// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

struct S3 {
  char Buffer[3];
};

struct S4 {
  char Buffer[4];
};

struct S15 {
  char Buffer[15];
};

struct S16 {
  char Buffer[16];
};

struct S127 {
  char Buffer[127];
};

struct S128 {
  char Buffer[128];
};

struct S3 g3;
struct S4 g4;
struct S15 g15;
struct S16 g16;
struct S127 g127;
struct S128 g128;

// CHECK: @g3 = global %struct.S3 zeroinitializer, align 1
// CHECK: @g4 = global %struct.S4 zeroinitializer, align 4
// CHECK: @g15 = global %struct.S15 zeroinitializer, align 4
// CHECK: @g16 = global %struct.S16 zeroinitializer, align 8
// CHECK: @g127 = global %struct.S127 zeroinitializer, align 8
// CHECK: @g128 = global %struct.S128 zeroinitializer, align 16
