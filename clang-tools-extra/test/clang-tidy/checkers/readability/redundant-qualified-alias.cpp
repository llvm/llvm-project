// RUN: %check_clang_tidy -std=c++11-or-later %s readability-redundant-qualified-alias %t
// RUN: %check_clang_tidy -check-suffix=NS -std=c++11-or-later %s readability-redundant-qualified-alias %t -- \
// RUN:   -config='{CheckOptions: { readability-redundant-qualified-alias.OnlyNamespaceScope: true }}'
// RUN: %check_clang_tidy -check-suffixes=,CXX23 -std=c++23 %s readability-redundant-qualified-alias %t
// RUN: %check_clang_tidy -check-suffixes=NS,NS-CXX23 -std=c++23 %s readability-redundant-qualified-alias %t -- \
// RUN:   -config='{CheckOptions: { readability-redundant-qualified-alias.OnlyNamespaceScope: true }}'

namespace n1 {
struct Foo {};
struct Bar {};
struct Attr {};
struct Commented {};
struct AfterType {};
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
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

using AliasDeprecated [[deprecated("alias attr")]] = n1::Foo;
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

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

using AfterType = n1::AfterType /*rhs-comment*/;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-MESSAGES-NS: :[[@LINE-2]]:7: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
// CHECK-FIXES: using n1::AfterType /*rhs-comment*/;

#define DECL_END ;
using MacroDeclEnd = n1::MacroType DECL_END
// CHECK-MESSAGES-NOT: warning: type alias is redundant; use a using-declaration instead
// CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead

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
};

void local_scope() {
  using LocalType = n1::LocalType;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: type alias is redundant; use a using-declaration instead [readability-redundant-qualified-alias]
  // CHECK-MESSAGES-NS-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-FIXES: using n1::LocalType;
}

#if __cplusplus >= 202302L
void cxx23_init_statement_scope(bool Cond) {
  if (using Foo = n1::Foo; Cond) {
  }
  // CHECK-MESSAGES-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-MESSAGES-NS-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead

  switch (using Bar = ::n1::Bar; 0) {
  default:
    break;
  }
  // CHECK-MESSAGES-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-MESSAGES-NS-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead

  for (using Deep = n2::n3::Deep; Cond;) {
    Cond = false;
  }
  // CHECK-MESSAGES-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-MESSAGES-NS-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead

  int Values[] = {0};
  for (using GlobalType = ::GlobalType; int V : Values) {
    (void)V;
  }
  // CHECK-MESSAGES-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead
  // CHECK-MESSAGES-NS-CXX23-NOT: warning: type alias is redundant; use a using-declaration instead
}
#endif
