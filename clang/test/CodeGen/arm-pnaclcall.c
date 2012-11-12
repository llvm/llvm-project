// RUN: %clang_cc1 -triple armv7-unknown-nacl-gnueabi \
// RUN:   -ffreestanding -mfloat-abi hard -target-cpu cortex-a8 \
// RUN:   -emit-llvm -w -o - %s | FileCheck %s

// Test that functions with pnaclcall attribute generate portable bitcode
// like the le32 arch target

typedef struct {
  int a;
  int b;
} s1;
// CHECK: define i32 @f48(%struct.s1* byval %s)
int __attribute__((pnaclcall)) f48(s1 s) { return s.a; }

// CHECK: define void @f49(%struct.s1* noalias sret %agg.result)
s1 __attribute__((pnaclcall)) f49() { s1 s; s.a = s.b = 1; return s; }

union simple_union {
  int a;
  char b;
};
// Unions should be passed as byval structs
// CHECK: define void @f50(%union.simple_union* byval %s)
void __attribute__((pnaclcall)) f50(union simple_union s) {}

typedef struct {
  int b4 : 4;
  int b3 : 3;
  int b8 : 8;
} bitfield1;
// Bitfields should be passed as byval structs
// CHECK: define void @f51(%struct.bitfield1* byval %bf1)
void __attribute__((pnaclcall)) f51(bitfield1 bf1) {}
