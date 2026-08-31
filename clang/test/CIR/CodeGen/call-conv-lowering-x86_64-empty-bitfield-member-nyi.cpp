// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

// `A` holds no data for the ABI yet occupies four bytes, which an unnamed
// bit-field access unit supplies.
struct A { unsigned : 32; };

struct AsBase : A { int i; };
void take_base(AsBase s) {}
// CHECK: not yet implemented for type '!cir.struct<"AsBase" {empty

struct AsMember { [[no_unique_address]] A a; int i; };
void take_member(AsMember s) {}
// CHECK: not yet implemented for type '!cir.struct<"AsMember" {empty
