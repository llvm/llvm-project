// RUN: %clang_cc1 -ast-print -std=c++2d %s | FileCheck %s

template <unsigned N, template <class> class... TT>
struct S {
  using type = TT...[N]<int>;
  using first = TT...[0]<int>;
};

// CHECK: template <unsigned int N, template <class> class ...TT> struct S {
// CHECK-NEXT: using type = TT...[N]<int>;
// CHECK-NEXT: using first = TT...[0]<int>;

template <template <class> class... TT>
TT...[0]<int> f(TT...[1]<int>);

// CHECK: template <template <class> class ...TT> TT...[0]<int> f(TT...[1]<int>);

template <template <class> concept... CC>
struct Constrained {
  template <CC...[0] T>
  static void f();
  static void g(CC...[1] auto);
};

// CHECK:      template <template <class> concept ...CC> struct Constrained {
// CHECK-NEXT: template <CC...[0] T> static void f();
// CHECK-NEXT: static void g(CC...[1] auto);

template <template <class> concept... CC>
constexpr bool concept_id = CC...[0]<int>;

// CHECK: template <template <class> concept ...CC> constexpr bool concept_id = CC...[0]<int>;

template <template <class> auto... VV>
constexpr int variable_template_id = VV...[1]<int>;

// CHECK: template <template <class> auto ...VV> constexpr int variable_template_id = VV...[1]<int>;

template <template <class> concept... CC>
void requires_clause() requires CC...[0]<int>;

// CHECK: template <template <class> concept ...CC> void requires_clause() requires CC...[0]<int>;
