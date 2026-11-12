// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s
//
// Sema-focused PTU rollback regression tests, restricted to complex class
// constructs. Each scenario is adapted from a real clang/test/SemaCXX case
// and, following that suite's own style, uses a genuine semantic error
// (deleted/implicitly-deleted special members, access-control violations,
// undefined partial specializations, ...) rather than a synthetic bad
// token wherever the construct naturally produces one. The failing chunk
// always pairs that real error with a legitimate new class on the same
// line, so a correct rollback must discard both together; the follow-up
// chunk re-exercises the same class machinery (SpecialMemberCache,
// access-control checks, partial-specialization matching, ...) to see
// whether the rollback left it consistent.

extern "C" int printf(const char *, ...);

// 1. Implicitly-deleted copy constructor via a non-copyable member, cf.
// SemaCXX/cxx11-call-to-deleted-constructor.cpp and
// SemaCXX/explicitly-defaulted.cpp. Computing "is this special member
// deleted" for Holder populates Sema::SpecialMemberCache.
struct NoCopy { NoCopy() = default; NoCopy(const NoCopy&) = delete; };
struct Holder { NoCopy nc; int tag; Holder(int t) : tag(t) {} };
Holder h1(1);
auto rc1 = printf("h1.tag=%d\n", h1.tag);
// CHECK: h1.tag=1

struct HolderBad { NoCopy nc; int tag; HolderBad(int t) : tag(t) {} }; Holder h_copy_bad = h1;
Holder h2(2);
auto rc2 = printf("h2.tag=%d\n", h2.tag);
// CHECK: h2.tag=2

// 2. Dependent-name member access inside a nested class template, cf.
// SemaCXX/dependent-types.cpp and SemaCXX/member-class-11.cpp. The bogus
// member is only diagnosed once Inner::scaled is instantiated for a
// concrete T, exercising two-phase lookup plus template instantiation
// bookkeeping (InstantiatingSpecializations, PendingInstantiations).
template<typename T> struct Outer {
  T value;
  struct Inner { T scaled(Outer<T>& o, T factor) { return o.value * factor; } };
};
Outer<int> outer1; outer1.value = 5;
Outer<int>::Inner inner1;
auto rc3 = printf("scaled=%d\n", inner1.scaled(outer1, 3));
// CHECK: scaled=15

template<typename T> struct OuterBad { T value; struct Inner { T scaled(OuterBad<T>& o, T factor) { return o.value * o.nonexistent_member * factor; } }; }; OuterBad<int> ob; ob.value = 1; OuterBad<int>::Inner ib; ib.scaled(ob, 2);
Outer<double> outer2; outer2.value = 2.5;
Outer<double>::Inner inner2;
auto rc4 = printf("scaled2=%.1f\n", inner2.scaled(outer2, 4.0));
// CHECK: scaled2=10.0

// 3. Protected-member access control across an unrelated class, cf.
// SemaCXX/access-control-check.cpp. ConvOutsider's direct access to
// ConvBase::secret is a genuine access-control error, alongside a
// legitimate new derived class on the same line.
struct ConvBase { protected: int secret = 10; public: int reveal() { return secret; } };
struct ConvDerived : ConvBase { int shout() { return secret * 2; } };
ConvDerived cd1;
auto rc5 = printf("reveal=%d shout=%d\n", cd1.reveal(), cd1.shout());
// CHECK: reveal=10 shout=20

struct ConvOutsider { int peek(ConvBase& b) { return b.secret; } }; void overload_pick(ConvDerived) {}
struct ConvDerived2 : ConvBase { int shout() { return secret * 3; } };
ConvDerived2 cd2;
auto rc6 = printf("reveal2=%d shout2=%d\n", cd2.reveal(), cd2.shout());
// CHECK: reveal2=10 shout2=30

// 4. Instantiating an undefined class-template partial specialization, cf.
// SemaCXX/undefined-partial-specialization.cpp and
// SemaCXX/identical-type-primary-partial-specialization.cpp.
template<typename T> struct Describe { static int tag() { return 0; } };
template<typename T> struct Describe<T*> { static int tag() { return 1; } };
auto rc7 = printf("tag_val=%d tag_ptr=%d\n", Describe<int>::tag(), Describe<int*>::tag());
// CHECK: tag_val=0 tag_ptr=1

template<typename T> struct Describe<T&>; template<typename T> struct DescribeBad { static int tag() { return Describe<T&>::tag(); } }; int bad_tag = DescribeBad<int>::tag();
auto rc8 = printf("tag_val2=%d tag_ptr2=%d\n", Describe<double>::tag(), Describe<double*>::tag());
// CHECK: tag_val2=0 tag_ptr2=1

// 5. Friend-granted access vs. a genuine private-access violation, cf.
// SemaCXX/friend.cpp and SemaCXX/friend-out-of-line.cpp.
struct FriendHolder {
private:
  int hidden = 42;
  friend int peek_hidden(FriendHolder&);
};
int peek_hidden(FriendHolder& f) { return f.hidden; }
FriendHolder fh1;
auto rc9 = printf("peek_hidden=%d\n", peek_hidden(fh1));
// CHECK: peek_hidden=42

struct FriendHolder2 { private: int hidden2 = 99; friend int peek_hidden2(FriendHolder2&); }; int bad_peek = fh1.hidden;
int peek_hidden2(FriendHolder2& f) { return f.hidden2; }
FriendHolder2 fh2;
auto rc10 = printf("peek_hidden2=%d\n", peek_hidden2(fh2));
// CHECK: peek_hidden2=99

// 6. Implicitly-deleted copy-assignment via a non-assignable member, cf.
// SemaCXX/deleted-function.cpp and SemaCXX/deleted-function-access.cpp.
struct NoAssign { int val; NoAssign(int v) : val(v) {} NoAssign& operator=(const NoAssign&) = delete; };
struct Container { NoAssign na; Container(int v) : na(v) {} };
Container c1(5);
auto rc11 = printf("c1.na.val=%d\n", c1.na.val);
// CHECK: c1.na.val=5

struct ContainerBad { NoAssign na; ContainerBad(int v) : na(v) {} }; Container c1b(9); c1 = c1b;
Container c2(11);
auto rc12 = printf("c2.na.val=%d\n", c2.na.val);
// CHECK: c2.na.val=11

%quit
