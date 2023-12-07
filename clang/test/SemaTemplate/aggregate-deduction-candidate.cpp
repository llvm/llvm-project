// RUN: %clang_cc1 -std=c++17 -verify=expected,cxx17 %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20 -ast-dump -ast-dump-decl-types -ast-dump-filter "deduction guide" %s | FileCheck %s --strict-whitespace

namespace Basic {
  template<class T> struct A { // cxx17-note 6 {{candidate}}
    T x;
    T y;
  };

  A a1 = {3.0, 4.0}; // cxx17-error {{no viable}}
  A a2 = {.x = 3.0, .y = 4.0}; // cxx17-error {{no viable}}

  A a3(3.0, 4.0); // cxx17-error {{no viable}}

  // CHECK-LABEL: Dumping Basic::<deduction guide for A>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for A>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced class depth 0 index 0 T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for A> 'auto (T, T) -> A<T>'
  // CHECK: | |-ParmVarDecl {{.*}} 'T'
  // CHECK: | `-ParmVarDecl {{.*}} 'T'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for A> 'auto (double, double) -> Basic::A<double>'
  // CHECK:   |-TemplateArgument type 'double'
  // CHECK:   | `-BuiltinType {{.*}} 'double'
  // CHECK:   |-ParmVarDecl {{.*}} 'double':'double'
  // CHECK:   `-ParmVarDecl {{.*}} 'double':'double'
  // CHECK: FunctionProtoType {{.*}} 'auto (T, T) -> A<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'A<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'A'
  // CHECK: |-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK: | `-TemplateTypeParm {{.*}} 'T'
  // CHECK: `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:   `-TemplateTypeParm {{.*}} 'T'

  template <typename T> struct S { // cxx20-note 2 {{candidate}}
    T x;
    T y;
  };

  template <typename T> struct C { // cxx20-note 10 {{candidate}} cxx17-note 12 {{candidate}}
    S<T> s;
    T t;
  };

  template <typename T> struct D { // cxx20-note 6 {{candidate}} cxx17-note 8 {{candidate}}
    S<int> s;
    T t;
  };

  C c1 = {1, 2}; // expected-error {{no viable}}
  C c2 = {1, 2, 3}; // expected-error {{no viable}}
  C c3 = {{1u, 2u}, 3}; // cxx17-error {{no viable}}

  C c4(1, 2);    // expected-error {{no viable}}
  C c5(1, 2, 3); // expected-error {{no viable}}
  C c6({1u, 2u}, 3); // cxx17-error {{no viable}}

  D d1 = {1, 2}; // expected-error {{no viable}}
  D d2 = {1, 2, 3}; // cxx17-error {{no viable}}

  D d3(1, 2); // expected-error {{no viable}}
  // CTAD succeed but brace elision is not allowed for parenthesized aggregate init. 
  D d4(1, 2, 3); // expected-error {{no viable}}

  // CHECK-LABEL: Dumping Basic::<deduction guide for C>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for C>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for C> 'auto (S<T>, T) -> C<T>'
  // CHECK: | |-ParmVarDecl {{.*}} 'S<T>':'S<T>'
  // CHECK: | `-ParmVarDecl {{.*}} 'T'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for C> 'auto (S<int>, int) -> Basic::C<int>'
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-BuiltinType {{.*}} 'int'
  // CHECK:   |-ParmVarDecl {{.*}} 'S<int>':'Basic::S<int>'
  // CHECK:   `-ParmVarDecl {{.*}} 'int':'int'
  // CHECK: FunctionProtoType {{.*}} 'auto (S<T>, T) -> C<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'C<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'C'
  // CHECK: |-ElaboratedType {{.*}} 'S<T>' sugar dependent
  // CHECK: | `-TemplateSpecializationType {{.*}} 'S<T>' dependent S
  // CHECK: |   `-TemplateArgument type 'T'
  // CHECK: |     `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK: |       `-TemplateTypeParm {{.*}} 'T'
  // CHECK: `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:   `-TemplateTypeParm {{.*}} 'T'

  // CHECK-LABEL: Dumping Basic::<deduction guide for D>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for D>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for D> 'auto (int, int) -> D<T>'
  // CHECK:   |-ParmVarDecl {{.*}} 'int':'int'
  // CHECK:   `-ParmVarDecl {{.*}} 'int':'int'
  // CHECK: FunctionProtoType {{.*}} 'auto (int, int) -> D<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'D<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'D'
  // CHECK: |-SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
  // CHECK: | |-ClassTemplateSpecialization {{.*}} 'S'
  // CHECK: | `-BuiltinType {{.*}} 'int'
  // CHECK: `-SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
  // CHECK:   |-ClassTemplateSpecialization {{.*}} 'S'
  // CHECK:   `-BuiltinType {{.*}} 'int'

  template <typename T> struct E { // cxx17-note 4 {{candidate}}
    T t;
    decltype(t) t2;
  };

  E e1 = {1, 2}; // cxx17-error {{no viable}}

  E e2(1, 2); // cxx17-error {{no viable}}

  // CHECK-LABEL: Dumping Basic::<deduction guide for E>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for E>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for E> 'auto (T, decltype(t)) -> E<T>'
  // CHECK: | |-ParmVarDecl {{.*}} 'T'
  // CHECK: | `-ParmVarDecl {{.*}} 'decltype(t)'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for E> 'auto (int, decltype(t)) -> Basic::E<int>'
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-BuiltinType {{.*}} 'int'
  // CHECK:   |-ParmVarDecl {{.*}} 'int':'int'
  // CHECK:   `-ParmVarDecl {{.*}} 'decltype(t)':'int'
  // CHECK: FunctionProtoType {{.*}} 'auto (T, decltype(t)) -> E<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'E<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'E'
  // CHECK: |-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK: | `-TemplateTypeParm {{.*}} 'T'
  // CHECK: `-DecltypeType {{.*}} 'decltype(t)' dependent
  // CHECK:   `-DeclRefExpr {{.*}} 'T' lvalue Field {{.*}} 't' 'T' non_odr_use_unevaluated

  template <typename T>
  struct I {
    using type = T;
  };

  template <typename T>
  struct F { // cxx17-note 2 {{candidate}}
    typename I<T>::type i;
    T t;
  };

  F f1 = {1, 2}; // cxx17-error {{no viable}}

  // CHECK-LABEL: Dumping Basic::<deduction guide for F>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for F>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for F> 'auto (typename I<T>::type, T) -> F<T>'
  // CHECK: | |-ParmVarDecl {{.*}} 'typename I<T>::type'
  // CHECK: | `-ParmVarDecl {{.*}} 'T'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for F> 'auto (typename I<int>::type, int) -> Basic::F<int>'
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-BuiltinType {{.*}} 'int'
  // CHECK:   |-ParmVarDecl {{.*}} 'typename I<int>::type':'int'
  // CHECK:   `-ParmVarDecl {{.*}} 'int':'int'
  // CHECK: FunctionProtoType {{.*}} 'auto (typename I<T>::type, T) -> F<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'F<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'F'
  // CHECK: |-DependentNameType {{.*}} 'typename I<T>::type' dependent
  // CHECK: `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:   `-TemplateTypeParm {{.*}} 'T'
}

namespace Array {
  typedef __SIZE_TYPE__ size_t;
  template <typename T, size_t N> struct A { // cxx20-note 2 {{candidate}} cxx17-note 14 {{candidate}}
    T array[N];
  };

  A a1 = {{1, 2, 3}}; // cxx17-error {{no viable}}
  A a2 = {1, 2, 3}; // expected-error {{no viable}}
  A a3 = {"meow"}; // cxx17-error {{no viable}}
  A a4 = {("meow")}; // cxx17-error {{no viable}}

  A a5({1, 2, 3}); // cxx17-error {{no viable}}
  A a6("meow"); // cxx17-error {{no viable}}
  A a7(("meow")); // cxx17-error {{no viable}}

  // CHECK-LABEL: Dumping Array::<deduction guide for A>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for A>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-NonTypeTemplateParmDecl {{.*}} 'size_t':'unsigned {{.*}}' depth 0 index 1 N
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for A> 'auto (T (&&)[N]) -> A<T, N>'
  // CHECK: | `-ParmVarDecl {{.*}} 'T (&&)[N]'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for A> 'auto (int (&&)[3]) -> Array::A<int, 3>'
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-BuiltinType {{.*}} 'int'
  // CHECK:   |-TemplateArgument integral 3
  // CHECK:   `-ParmVarDecl {{.*}} 'int (&&)[3]'
  // CHECK: FunctionProtoType {{.*}} 'auto (T (&&)[N]) -> A<T, N>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'A<T, N>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'A'
  // CHECK: `-RValueReferenceType {{.*}} 'T (&&)[N]' dependent
  // CHECK:   `-DependentSizedArrayType {{.*}} 'T[N]' dependent
  // CHECK:     |-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:     | `-TemplateTypeParm {{.*}} 'T'
  // CHECK:     `-DeclRefExpr {{.*}} 'size_t':'unsigned {{.*}}' NonTypeTemplateParm {{.*}} 'N' 'size_t':'unsigned {{.*}}'

  // CHECK: Dumping Array::<deduction guide for A>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for A>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-NonTypeTemplateParmDecl {{.*}} 'size_t':'unsigned {{.*}}' depth 0 index 1 N
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for A> 'auto (const T (&)[N]) -> A<T, N>'
  // CHECK: | `-ParmVarDecl {{.*}} 'const T (&)[N]'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for A> 'auto (const char (&)[5]) -> Array::A<char, 5>'
  // CHECK:   |-TemplateArgument type 'char'
  // CHECK:   | `-BuiltinType {{.*}} 'char'
  // CHECK:   |-TemplateArgument integral 5
  // CHECK:   `-ParmVarDecl {{.*}} 'const char (&)[5]'
  // CHECK: FunctionProtoType {{.*}} 'auto (const T (&)[N]) -> A<T, N>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'A<T, N>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'A'
  // CHECK: `-LValueReferenceType {{.*}} 'const T (&)[N]' dependent
  // CHECK:   `-QualType {{.*}} 'const T[N]' const
  // CHECK:     `-DependentSizedArrayType {{.*}} 'T[N]' dependent
  // CHECK:       |-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:       | `-TemplateTypeParm {{.*}} 'T'
  // CHECK:       `-DeclRefExpr {{.*}} 'size_t':'unsigned{{.*}}' NonTypeTemplateParm {{.*}} 'N' 'size_t':'unsigned{{.*}}'
}

namespace BraceElision {
  template <typename T> struct A { // cxx17-note 4 {{candidate}}
    T array[2];
  };

  A a1 = {0, 1}; // cxx17-error {{no viable}}

  // CTAD succeed but brace elision is not allowed for parenthesized aggregate init. 
  A a2(0, 1); // cxx20-error {{array initializer must be an initializer list}} cxx17-error {{no viable}}

  // CHECK-LABEL: Dumping BraceElision::<deduction guide for A>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for A>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for A> 'auto (T, T) -> A<T>'
  // CHECK: | |-ParmVarDecl {{.*}} 'T'
  // CHECK: | `-ParmVarDecl {{.*}} 'T'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for A> 'auto (int, int) -> BraceElision::A<int>'
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-BuiltinType {{.*}} 'int'
  // CHECK:   |-ParmVarDecl {{.*}} 'int':'int'
  // CHECK:   `-ParmVarDecl {{.*}} 'int':'int'
  // CHECK: FunctionProtoType {{.*}} 'auto (T, T) -> A<T>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'A<T>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'A'
  // CHECK: |-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK: | `-TemplateTypeParm {{.*}} 'T'
  // CHECK: `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
  // CHECK:   `-TemplateTypeParm {{.*}} 'T'
}

namespace TrailingPack {
  template<typename... T> struct A : T... { // cxx17-note 4 {{candidate}}
  };

  A a1 = { // cxx17-error {{no viable}}
    []{ return 1; },
    []{ return 2; }
  };

  A a2( // cxx17-error {{no viable}}
    []{ return 1; },
    []{ return 2; }
  );

  // CHECK-LABEL: Dumping TrailingPack::<deduction guide for A>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for A>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 ... T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for A> 'auto (T...) -> A<T...>'
  // CHECK: | `-ParmVarDecl {{.*}} 'T...' pack
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for A> 
  // CHECK-SAME: 'auto (TrailingPack::(lambda at {{.*}}), TrailingPack::(lambda at {{.*}})) -> 
  // CHECK-SAME:     TrailingPack::A<TrailingPack::(lambda at {{.*}}), TrailingPack::(lambda at {{.*}})>'
  // CHECK: |-TemplateArgument pack
  // CHECK: | |-TemplateArgument type 'TrailingPack::(lambda at {{.*}})'
  // CHECK: | | `-RecordType {{.*}} 'TrailingPack::(lambda at {{.*}})'
  // CHECK: | |   `-CXXRecord {{.*}} ''
  // CHECK: | `-TemplateArgument type 'TrailingPack::(lambda at {{.*}})'
  // CHECK: |   `-RecordType {{.*}} 'TrailingPack::(lambda at {{.*}})'
  // CHECK: |     `-CXXRecord {{.*}} ''
  // CHECK: |-ParmVarDecl {{.*}} 'TrailingPack::(lambda at {{.*}})':'TrailingPack::(lambda at {{.*}})'
  // CHECK: `-ParmVarDecl {{.*}} 'TrailingPack::(lambda at {{.*}})':'TrailingPack::(lambda at {{.*}})'
  // CHECK: FunctionProtoType {{.*}} 'auto (T...) -> A<T...>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'A<T...>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'A'
  // CHECK: `-PackExpansionType {{.*}} 'T...' dependent
  // CHECK:   `-TemplateTypeParmType {{.*}} 'T' dependent contains_unexpanded_pack depth 0 index 0 pack
  // CHECK:     `-TemplateTypeParm {{.*}} 'T'
}

namespace NonTrailingPack {
  template<typename... T> struct A : T... { // expected-note 4 {{candidate}}
    int a;
  };

  A a1 = { // expected-error {{no viable}}
    []{ return 1; },
    []{ return 2; }
  };

  A a2( // expected-error {{no viable}}
    []{ return 1; },
    []{ return 2; }
  );
}

namespace DeduceArity {
  template <typename... T> struct Types {};
  template <typename... T> struct F : Types<T...>, T... {}; // cxx20-note 12 {{candidate}} cxx17-note 16 {{candidate}}

  struct X {};
  struct Y {};
  struct Z {};
  struct W { operator Y(); };

  F f1 = {Types<X, Y, Z>{}, {}, {}}; // cxx17-error {{no viable}}
  F f2 = {Types<X, Y, Z>{}, X{}, Y{}}; // cxx17-error {{no viable}}
  F f3 = {Types<X, Y, Z>{}, X{}, W{}}; // expected-error {{no viable}}
  F f4 = {Types<X>{}, {}, {}}; // expected-error {{no viable}}

  F f5(Types<X, Y, Z>{}, {}, {}); // cxx17-error {{no viable}}
  F f6(Types<X, Y, Z>{}, X{}, Y{}); // cxx17-error {{no viable}}
  F f7(Types<X, Y, Z>{}, X{}, W{}); // expected-error {{no viable}}
  F f8(Types<X>{}, {}, {}); // expected-error {{no viable}}

  // CHECK-LABEL: Dumping DeduceArity::<deduction guide for F>:
  // CHECK: FunctionTemplateDecl {{.*}} implicit <deduction guide for F>
  // CHECK: |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 ... T
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for F> 'auto (Types<T...>, T...) -> F<T...>'
  // CHECK: | |-ParmVarDecl {{.*}} 'Types<T...>':'Types<T...>'
  // CHECK: | `-ParmVarDecl {{.*}} 'T...' pack
  // CHECK: |-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for F> 
  // CHECK-SAME: 'auto (Types<X, Y, Z>, DeduceArity::X, DeduceArity::Y, DeduceArity::Z) -> 
  // CHECK-SAME:     DeduceArity::F<DeduceArity::X, DeduceArity::Y, DeduceArity::Z>'
  // CHECK: | |-TemplateArgument pack
  // CHECK: | | |-TemplateArgument type 'DeduceArity::X'
  // CHECK: | | | `-RecordType {{.*}} 'DeduceArity::X'
  // CHECK: | | |   `-CXXRecord {{.*}} 'X'
  // CHECK: | | |-TemplateArgument type 'DeduceArity::Y'
  // CHECK: | | | `-RecordType {{.*}} 'DeduceArity::Y'
  // CHECK: | | |   `-CXXRecord {{.*}} 'Y'
  // CHECK: | | `-TemplateArgument type 'DeduceArity::Z'
  // CHECK: | |   `-RecordType {{.*}} 'DeduceArity::Z'
  // CHECK: | |     `-CXXRecord {{.*}} 'Z'
  // CHECK: | |-ParmVarDecl {{.*}} 'Types<X, Y, Z>':'DeduceArity::Types<DeduceArity::X, DeduceArity::Y, DeduceArity::Z>'
  // CHECK: | |-ParmVarDecl {{.*}} 'DeduceArity::X':'DeduceArity::X'
  // CHECK: | |-ParmVarDecl {{.*}} 'DeduceArity::Y':'DeduceArity::Y'
  // CHECK: | `-ParmVarDecl {{.*}} 'DeduceArity::Z':'DeduceArity::Z'
  // CHECK: `-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for F> 'auto (Types<X>, DeduceArity::X) -> DeduceArity::F<DeduceArity::X>'
  // CHECK:   |-TemplateArgument pack
  // CHECK:   | `-TemplateArgument type 'DeduceArity::X'
  // CHECK:   |   `-RecordType {{.*}} 'DeduceArity::X'
  // CHECK:   |     `-CXXRecord {{.*}} 'X'
  // CHECK:   |-ParmVarDecl {{.*}} 'Types<X>':'DeduceArity::Types<DeduceArity::X>'
  // CHECK:   `-ParmVarDecl {{.*}} 'DeduceArity::X':'DeduceArity::X'
  // CHECK: FunctionProtoType {{.*}} 'auto (Types<T...>, T...) -> F<T...>' dependent trailing_return cdecl
  // CHECK: |-InjectedClassNameType {{.*}} 'F<T...>' dependent
  // CHECK: | `-CXXRecord {{.*}} 'F'
  // CHECK: |-ElaboratedType {{.*}} 'Types<T...>' sugar dependent
  // CHECK: | `-TemplateSpecializationType {{.*}} 'Types<T...>' dependent Types
  // CHECK: |   `-TemplateArgument type 'T...'
  // CHECK: |     `-PackExpansionType {{.*}} 'T...' dependent
  // CHECK: |       `-TemplateTypeParmType {{.*}} 'T' dependent contains_unexpanded_pack depth 0 index 0 pack
  // CHECK: |         `-TemplateTypeParm {{.*}} 'T'
  // CHECK: `-PackExpansionType {{.*}} 'T...' dependent
  // CHECK:   `-TemplateTypeParmType {{.*}} 'T' dependent contains_unexpanded_pack depth 0 index 0 pack
  // CHECK:     `-TemplateTypeParm {{.*}} 'T'
}
