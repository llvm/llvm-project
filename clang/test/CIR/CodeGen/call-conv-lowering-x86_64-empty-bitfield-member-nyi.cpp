// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - 2>&1 | FileCheck %s

// `A` holds no data for the ABI, yet it occupies four bytes, because an
// unnamed bit-field access unit supplies them.  Reached as a member it takes
// the same `empty` mark a unit does, so classifying it as one would count
// bytes classic CodeGen counts through the member's own fields.
struct A { unsigned : 32; };

struct AsBase : A { int i; };
void take_base(AsBase s) {}
// CHECK: not yet implemented for type '!cir.struct<"AsBase" {empty

struct AsMember { [[no_unique_address]] A a; int i; };
void take_member(AsMember s) {}
// CHECK: not yet implemented for type '!cir.struct<"AsMember" {empty
