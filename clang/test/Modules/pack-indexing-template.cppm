// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++2d -triple %itanium_abi_triple -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++2d -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm -emit-llvm -o - %t/b.cpp | FileCheck %s
// RUN: %clang_cc1 -std=c++2d -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm -fsyntax-only -verify %t/c.cpp

//--- a.cppm
export module a;

template <class T> struct Wrapper {
  using type = T;
  T value;
};
template <class T> struct Other {};

export template <unsigned I, template <class> class... TT>
using Indexed = TT...[I]<int>;

export template <template <class> class... TT>
struct Holder {
  using first = TT...[0]<int>;
  typename TT...[0]<int>::type value;
};

export template <class T> concept Always = true;
export template <class T> concept Never = false;
export template <class T> constexpr int Var = 1;
export template <class T> constexpr int Var2 = 2;

export template <unsigned I, template <class> concept... CC>
constexpr bool ConceptId = CC...[I]<int>;

export template <unsigned I, template <class> auto... VV>
constexpr int VariableTemplateId = VV...[I]<int>;

export template <template <class> concept... CC>
struct Constrained {
  template <CC...[0] T>
  static constexpr int f() { return 3; }
  static constexpr int g(CC...[0] auto) { return 4; }
};

export Indexed<0, Wrapper, Other> a = {42};

//--- b.cpp
import a;

int b() {
  return a.value;
}

// CHECK: @_ZW1a1a = external global %struct.Wrapper
// CHECK: define {{.*}}i32 @_Z1bv()

//--- c.cpp
// expected-no-diagnostics
import a;

template <class T> struct Local {
  using type = T;
};
template <class T> struct Unused {};

static_assert(__is_same(Indexed<1, Local, Unused>, Unused<int>));
static_assert(__is_same(Holder<Local>::first, Local<int>));
static_assert(__is_same(decltype(Holder<Local>::value), int));

static_assert(ConceptId<0, Always, Never>);
static_assert(!ConceptId<1, Always, Never>);
static_assert(VariableTemplateId<1, Var, Var2> == 2);
static_assert(Constrained<Always>::f<int>() == 3);
static_assert(Constrained<Always>::g(0) == 4);
