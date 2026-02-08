// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-qualified-alias %t
// RUN: %check_clang_tidy -check-suffix=NS -std=c++11-or-later %s readability-redundant-qualified-alias %t -- \
// RUN:   -config='{CheckOptions: { readability-redundant-qualified-alias.OnlyNamespaceScope: true }}'

namespace n1 {
struct Foo {};
struct Bar {};
struct Attr {};
struct Commented {};
struct Elab {};
struct MacroEq {};
struct MacroType {};
struct PtrType {};
struct LocalType {};
} // namespace n1

namespace n2 {
namespace n3 {
struct Deep {};
} // namespace n3
} // namespace n2

namespace td {
typedef n1::Foo TypedefFoo;
} // namespace td

struct GlobalType {};

using Foo = n1::Foo;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using n1::Foo;

using Bar = ::n1::Bar;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using ::n1::Bar;

using Attr = n1::Attr __attribute__((aligned(8)));
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using n1::Attr __attribute__((aligned(8)));

using Deep = n2::n3::Deep;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using n2::n3::Deep;

using TypedefFoo = td::TypedefFoo;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using td::TypedefFoo;

using GlobalType = ::GlobalType;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using ::GlobalType;

using Builtin = int;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

using PtrType = n1::PtrType *;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

namespace templ {
template <typename T>
struct Vec {};
} // namespace templ

using Vec = templ::Vec<int>;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

namespace templ_alias {
template <typename T>
using Foo = n1::Foo;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead
} // namespace templ_alias

template <typename T>
struct Dependent {
  using X = typename T::X;
  // CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead
};

using Elab = class n1::Elab;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

using Commented /*comment*/ = n1::Commented;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using Commented /*comment*/ = n1::Commented;
#define ALIAS MacroType
using ALIAS = n1::MacroType;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

#define RHS n1::MacroType
using MacroType = RHS;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

#define EQ =
using MacroEq EQ n1::MacroEq;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

struct Base {
  using T = n1::Foo;
};

struct Derived : Base {
  using T = Base::T;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
  // CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-FIXES: using Base::T;
  // CHECK-FIXES-NS: using T = Base::T;
};

void local_scope() {
  using LocalType = n1::LocalType;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
  // CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-FIXES: using n1::LocalType;
  // CHECK-FIXES-NS: using LocalType = n1::LocalType;
}
