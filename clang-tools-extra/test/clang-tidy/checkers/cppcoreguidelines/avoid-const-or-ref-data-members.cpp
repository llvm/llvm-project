// RUN: %check_clang_tidy %s cppcoreguidelines-avoid-const-or-ref-data-members %t
namespace std {
template <typename T>
struct unique_ptr {};

template <typename T>
struct shared_ptr {};
} // namespace std

namespace gsl {
template <typename T>
struct not_null {};
} // namespace gsl

struct Ok {
  int i;
  int *p;
  const int *pc;
  std::unique_ptr<int> up;
  std::shared_ptr<int> sp;
  gsl::not_null<int> n;
};

struct ConstMember {
  const int c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const int' is const qualified [cppcoreguidelines-avoid-const-or-ref-data-members]
};

struct LvalueRefMember {
  int &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'int &' is a reference
};

struct ConstRefMember {
  const int &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const int &' is a reference
};

struct RvalueRefMember {
  int &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'int &&' is a reference
};

struct ConstAndRefMembers {
  const int c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const int' is const qualified
  int &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'int &' is a reference
  const int &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const int &' is a reference
  int &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'int &&' is a reference
};

struct Foo {};

struct Ok2 {
  Foo i;
  Foo *p;
  const Foo *pc;
  std::unique_ptr<Foo> up;
  std::shared_ptr<Foo> sp;
  gsl::not_null<Foo> n;
};

struct ConstMember2 {
  const Foo c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const Foo' is const qualified
};

struct LvalueRefMember2 {
  Foo &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'Foo &' is a reference
};

struct ConstRefMember2 {
  const Foo &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const Foo &' is a reference
};

struct RvalueRefMember2 {
  Foo &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'Foo &&' is a reference
};

struct ConstAndRefMembers2 {
  const Foo c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'const Foo' is const qualified
  Foo &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'lr' of type 'Foo &' is a reference
  const Foo &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'cr' of type 'const Foo &' is a reference
  Foo &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: member 'rr' of type 'Foo &&' is a reference
};

using ConstType = const int;
using RefType = int &;
using ConstRefType = const int &;
using RefRefType = int &&;

struct WithAlias {
  ConstType c;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'c' of type 'ConstType' (aka 'const int') is const qualified
  RefType lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: member 'lr' of type 'RefType' (aka 'int &') is a reference
  ConstRefType cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: member 'cr' of type 'ConstRefType' (aka 'const int &') is a reference
  RefRefType rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'rr' of type 'RefRefType' (aka 'int &&') is a reference
};

template <int N>
using Array = int[N];

struct ConstArrayMember {
  const Array<1> c;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: member 'c' of type 'const Array<1>' (aka 'const int[1]') is const qualified
};

struct LvalueRefArrayMember {
  Array<2> &lr;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: member 'lr' of type 'Array<2> &' (aka 'int (&)[2]') is a reference
};

struct ConstLvalueRefArrayMember {
  const Array<3> &cr;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: member 'cr' of type 'const Array<3> &' (aka 'const int (&)[3]') is a reference
};

struct RvalueRefArrayMember {
  Array<4> &&rr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: member 'rr' of type 'Array<4> &&' (aka 'int (&&)[4]') is a reference
};

template <typename T>
struct TemplatedOk {
  T t;
};

template <typename T>
struct TemplatedConst {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'const int' is const qualified
};

template <typename T>
struct TemplatedConstRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'const int &' is a reference
};

template <typename T>
struct TemplatedRefRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'int &&' is a reference
};

template <typename T>
struct TemplatedRef {
  T t;
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: member 't' of type 'int &' is a reference
};

TemplatedOk<int> t1{};
TemplatedConst<const int> t2{123};
TemplatedConstRef<const int &> t3{123};
TemplatedRefRef<int &&> t4{123};
TemplatedRef<int &> t5{t1.t};

// Lambdas capturing const or ref members should not trigger warnings
void lambdas()
{
  int x1{123};
  const int x2{123};
  const int& x3{123};
  int&& x4{123};
  int& x5{x1};

  auto v1 = [x1]{};
  auto v2 = [x2]{};
  auto v3 = [x3]{};
  auto v4 = [x4]{};
  auto v5 = [x5]{};

  auto r1 = [&x1]{};
  auto r2 = [&x2]{};
  auto r3 = [&x3]{};
  auto r4 = [&x4]{};
  auto r5 = [&x5]{};

  auto iv = [=]{
    auto c1 = x1;
    auto c2 = x2;
    auto c3 = x3;
    auto c4 = x4;
    auto c5 = x5;
  };

  auto ir = [&]{
    auto c1 = x1;
    auto c2 = x2;
    auto c3 = x3;
    auto c4 = x4;
    auto c5 = x5;
  };
}

struct NonCopyableWithRef
{
  NonCopyableWithRef(NonCopyableWithRef const&) = delete;
  NonCopyableWithRef& operator=(NonCopyableWithRef const&) = delete;
  NonCopyableWithRef(NonCopyableWithRef&&) = default;
  NonCopyableWithRef& operator=(NonCopyableWithRef&&) = default;

  int& x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'x' of type 'int &' is a reference
};

struct NonMovableWithRef
{
  NonMovableWithRef(NonMovableWithRef const&) = default;
  NonMovableWithRef& operator=(NonMovableWithRef const&) = default;
  NonMovableWithRef(NonMovableWithRef&&) = delete;
  NonMovableWithRef& operator=(NonMovableWithRef&&) = delete;

  int& x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'x' of type 'int &' is a reference
};

struct NonCopyableNonMovableWithRef
{
  NonCopyableNonMovableWithRef(NonCopyableNonMovableWithRef const&) = delete;
  NonCopyableNonMovableWithRef(NonCopyableNonMovableWithRef&&) = delete;
  NonCopyableNonMovableWithRef& operator=(NonCopyableNonMovableWithRef const&) = delete;
  NonCopyableNonMovableWithRef& operator=(NonCopyableNonMovableWithRef&&) = delete;

  int& x; // OK, non copyable nor movable
};

struct NonCopyable
{
  NonCopyable(NonCopyable const&) = delete;
  NonCopyable& operator=(NonCopyable const&) = delete;
  NonCopyable(NonCopyable&&) = default;
  NonCopyable& operator=(NonCopyable&&) = default;
};

struct NonMovable
{
  NonMovable(NonMovable const&) = default;
  NonMovable& operator=(NonMovable const&) = default;
  NonMovable(NonMovable&&) = delete;
  NonMovable& operator=(NonMovable&&) = delete;
};

struct NonCopyableNonMovable
{
  NonCopyableNonMovable(NonCopyableNonMovable const&) = delete;
  NonCopyableNonMovable(NonCopyableNonMovable&&) = delete;
  NonCopyableNonMovable& operator=(NonCopyableNonMovable const&) = delete;
  NonCopyableNonMovable& operator=(NonCopyableNonMovable&&) = delete;
};

// Test inheritance
struct InheritFromNonCopyable : NonCopyable
{
  int& x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'x' of type 'int &' is a reference
};

struct InheritFromNonMovable : NonMovable
{
  int& x;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'x' of type 'int &' is a reference
};

struct InheritFromNonCopyableNonMovable : NonCopyableNonMovable
{
  int& x;  // OK, non copyable nor movable
};

struct InheritBothFromNonCopyableAndNonMovable : NonCopyable, NonMovable
{
  int& x;  // OK, non copyable nor movable
};

// Test composition
struct ContainsNonCopyable
{
  NonCopyable x;
  int& y;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'y' of type 'int &' is a reference
};

struct ContainsNonMovable
{
  NonMovable x;
  int& y;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: member 'y' of type 'int &' is a reference
};

struct ContainsNonCopyableNonMovable
{
  NonCopyableNonMovable x;
  int& y;  // OK, non copyable nor movable
};

struct ContainsBothNonCopyableAndNonMovable
{
  NonCopyable x;
  NonMovable y;
  int& z;  // OK, non copyable nor movable
};

// If copies are deleted and moves are not declared, moves are not implicitly declared,
// so the class is also not movable and we should not warn
struct NonCopyableMovesNotDeclared
{
  NonCopyableMovesNotDeclared(NonCopyableMovesNotDeclared const&) = delete;
  NonCopyableMovesNotDeclared& operator=(NonCopyableMovesNotDeclared const&) = delete;

  int& x;  // OK, non copyable nor movable
};

// If moves are deleted but copies are not declared, copies are implicitly deleted,
// so the class is also not copyable and we should not warn
struct NonMovableCopiesNotDeclared
{
  NonMovableCopiesNotDeclared(NonMovableCopiesNotDeclared&&) = delete;
  NonMovableCopiesNotDeclared& operator=(NonMovableCopiesNotDeclared&&) = delete;

  int& x;  // OK, non copyable nor movable
};
