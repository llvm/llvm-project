// Tests that lifetime-extended temporaries whose backing storage is a
// subobject (member or base) are always emitted as `internal global`, never
// 'internal constant', regardless of the cv-qualification on the reference.
//
// In C++98, skipRValueSubobjectAdjustments is used for rvalue subobject
// adjustments.  The MaterializeTemporaryExpr ends up with the type of the
// reference (e.g. `const int`), not the type of the backing store (e.g. `S`).
// hasSameUnqualifiedType detects this mismatch and correctly falls back to
// Init->getType(), preventing the backing store from being marked constant.
//
// RUN: %clang_cc1 -std=c++98 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

// `const int &` bound to a member of an S temporary.
// The backing store is the whole S object, which can't be stored const.

struct MemberS { int x, y; };
const int &member_ref = MemberS().x;

// CHECK: @_ZGR10member_ref_ = internal global %struct.MemberS

// Binding a `const int &` to a mutable member.
// The backing store is the whole object (with a mutable member), so it 
// must remain writable.

struct MutableS {
  mutable int m;
  MutableS() : m(1) {}
};
const int &mutable_member_ref = MutableS().m;
void write_mutable() { const_cast<int &>(mutable_member_ref) = 5; }

// CHECK: @_ZGR18mutable_member_ref_ = internal global %struct.MutableS

// `const Base &` bound to a Derived temporary.
// Non-constexpr constructors mean no constant initializer is possible, and
// the backing store is the full Derived object.

struct Base { int b; Base() : b(11) {} };
struct Derived : Base { int d; Derived() : Base(), d(22) {} };
const Base &base_ref = Derived();

// CHECK: @_ZGR8base_ref_ = internal global %struct.Derived

// Same as above but using a plain member instead of a base class.

struct Pair { int a, c; Pair() : a(33), c(44) {} };
const int &pair_member_ref = Pair().a;

// CHECK: @_ZGR15pair_member_ref_ = internal global %struct.Pair
