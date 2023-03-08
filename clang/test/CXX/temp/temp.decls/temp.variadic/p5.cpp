// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++11 -fblocks -fms-extensions -fsyntax-only -verify=expected,cxx11 %s
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -std=c++2b -fblocks -fms-extensions -fsyntax-only -verify=expected %s

template<typename T, typename U> struct pair;
template<typename ...> struct tuple;

// A parameter pack whose name appears within the pattern of a pack
// expansion is expanded by that pack expansion. An appearance of the
// name of a parameter pack is only expanded by the innermost
// enclosing pack expansion. The pattern of a pack expansion shall
// name one or more parameter packs that are not expanded by a nested
// pack expansion.
template<typename... Types>
struct Expansion {
  typedef pair<Types..., int> expand_with_pacs; // okay
  typedef pair<Types, int...> expand_no_packs;  // expected-error{{pack expansion does not contain any unexpanded parameter packs}}
  typedef pair<pair<Types..., int>..., int> expand_with_expanded_nested; // expected-error{{pack expansion does not contain any unexpanded parameter packs}}
};

// All of the parameter packs expanded by a pack expansion shall have
// the same number of arguments specified.
template<typename ...Types>
struct ExpansionLengthMismatch {
  template<typename ...OtherTypes>
  struct Inner {
    typedef tuple<pair<Types, OtherTypes>...> type; // expected-error{{pack expansion contains parameter packs 'Types' and 'OtherTypes' that have different lengths (3 vs. 2)}}
  };
};

ExpansionLengthMismatch<int, long>::Inner<unsigned int, unsigned long>::type 
  *il_pairs;
tuple<pair<int, unsigned int>, pair<long, unsigned long> >*il_pairs_2 = il_pairs;

ExpansionLengthMismatch<short, int, long>::Inner<unsigned int, unsigned long>::type // expected-note{{in instantiation of template class 'ExpansionLengthMismatch<short, int, long>::Inner<unsigned int, unsigned long>' requested here}}
  *il_pairs_bad; 


// An appearance of a name of a parameter pack that is not expanded is
// ill-formed.

// Test for unexpanded parameter packs in each of the type nodes.
template<typename T, int N, typename ... Types>
struct TestPPName 
  : public Types, public T  // expected-error{{base type contains unexpanded parameter pack 'Types'}}
{
  // BuiltinType is uninteresting
  // FIXME: ComplexType is uninteresting?
  // PointerType
  typedef Types *types_pointer; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}

  // BlockPointerType
  typedef Types (^block_pointer_1)(int); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
  typedef int (^block_pointer_2)(Types); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}

  // LValueReferenceType
  typedef Types &lvalue_ref; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}

  // RValueReferenceType
  typedef Types &&rvalue_ref; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}

  // MemberPointerType
  typedef Types TestPPName::* member_pointer_1; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
  typedef int Types::*member_pointer_2; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // ConstantArrayType
  typedef Types constant_array[17]; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // IncompleteArrayType
  typedef Types incomplete_array[]; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // VariableArrayType
  void f(int i) {
    Types variable_array[i]; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 
  }

  // DependentSizedArrayType
  typedef Types dependent_sized_array[N]; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // DependentSizedExtVectorType
  typedef Types dependent_sized_ext_vector __attribute__((ext_vector_type(N))); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // VectorType is uninteresting

  // ExtVectorType
  typedef Types ext_vector __attribute__((ext_vector_type(4))); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // FunctionProtoType
  typedef Types (function_type_1)(int); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 
  typedef int (function_type_2)(Types); // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // FunctionNoProtoType is uninteresting
  // UnresolvedUsingType is uninteresting
  // ParenType is uninteresting
  // TypedefType is uninteresting

  // TypeOfExprType
  typedef __typeof__((static_cast<Types>(0))) typeof_expr; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // TypeOfType
  typedef __typeof__(Types) typeof_type;  // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // DecltypeType
  typedef decltype((static_cast<Types>(0))) typeof_expr; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // RecordType is uninteresting
  // EnumType is uninteresting
  // ElaboratedType is uninteresting

  // TemplateTypeParmType
  typedef Types template_type_parm; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // SubstTemplateTypeParmType is uninteresting

  // TemplateSpecializationType
  typedef pair<Types, int> template_specialization; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // InjectedClassName is uninteresting.

  // DependentNameType
  typedef typename Types::type dependent_name; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // DependentTemplateSpecializationType
  typedef typename Types::template apply<int> dependent_name_1; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 
  typedef typename T::template apply<Types> dependent_name_2; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}} 

  // ObjCObjectType is uninteresting
  // ObjCInterfaceType is uninteresting
  // ObjCObjectPointerType is uninteresting
};

// FIXME: Test for unexpanded parameter packs in each of the expression nodes.
template<int ...Values>
void test_unexpanded_in_exprs() {
  // PredefinedExpr is uninteresting
  // DeclRefExpr
  Values; // expected-error{{expression contains unexpanded parameter pack 'Values'}}
  // IntegerLiteral is uninteresting
  // FloatingLiteral is uninteresting
  // ImaginaryLiteral is uninteresting
  // StringLiteral is uninteresting
  // CharacterLiteral is uninteresting
  (Values); // expected-error{{expression contains unexpanded parameter pack 'Values'}}
  // UnaryOperator
  -Values; // expected-error{{expression contains unexpanded parameter pack 'Values'}}
  // OffsetOfExpr
  struct OffsetMe {
    int array[17];
  };
  __builtin_offsetof(OffsetMe, array[Values]); // expected-error{{expression contains unexpanded parameter pack 'Values'}}
  // FIXME: continue this...
}

template<typename ... Types>
void TestPPNameFunc(int i) {
  f(static_cast<Types>(i)); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
}

template<typename T, template<class> class ...Meta>
struct TestUnexpandedTTP {
  typedef tuple<typename Meta<T>::type> type; // expected-error{{declaration type contains unexpanded parameter pack 'Meta'}}
};

// Test for unexpanded parameter packs in declarations.
template<typename T, typename... Types>
// FIXME: this should test that the diagnostic reads "type contains..."
struct alignas(Types) TestUnexpandedDecls : T{ // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  void member_function(Types);  // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
#if __cplusplus < 201703L
  void member_function () throw(Types); // expected-error{{exception type contains unexpanded parameter pack 'Types'}}
#endif
  void member_function2() noexcept(Types()); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  operator Types() const; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
  Types data_member;  // expected-error{{data member type contains unexpanded parameter pack 'Types'}}
  static Types static_data_member; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
  unsigned bit_field : static_cast<Types>(0);  // expected-error{{bit-field size contains unexpanded parameter pack 'Types'}}
  static_assert(static_cast<Types>(0), "Boom"); // expected-error{{static assertion contains unexpanded parameter pack 'Types'}}

  enum E0 : Types {  // expected-error{{fixed underlying type contains unexpanded parameter pack 'Types'}}
    EnumValue = static_cast<Types>(0) // expected-error{{enumerator value contains unexpanded parameter pack 'Types'}}
  };

  using typename Types::type; // expected-error{{using declaration contains unexpanded parameter pack 'Types'}}
  using Types::value; // expected-error{{using declaration contains unexpanded parameter pack 'Types'}}
  using T::operator Types; // expected-error{{using declaration contains unexpanded parameter pack 'Types'}}

  friend class Types::foo; // expected-error{{friend declaration contains unexpanded parameter pack 'Types'}}
  friend void friend_func(Types); // expected-error{{friend declaration contains unexpanded parameter pack 'Types'}}
  friend void Types::other_friend_func(int); // expected-error{{friend declaration contains unexpanded parameter pack 'Types'}}

  void test_initializers() {
    T copy_init = static_cast<Types>(0); // expected-error{{initializer contains unexpanded parameter pack 'Types'}}
    T direct_init(0, static_cast<Types>(0)); // expected-error{{initializer contains unexpanded parameter pack 'Types'}}
    T list_init = { static_cast<Types>(0) }; // expected-error{{initializer contains unexpanded parameter pack 'Types'}}
  }

  T in_class_member_init = static_cast<Types>(0); // expected-error{{initializer contains unexpanded parameter pack 'Types'}}
  TestUnexpandedDecls() : 
    Types(static_cast<Types>(0)), // expected-error{{initializer contains unexpanded parameter pack 'Types'}}
    Types(static_cast<Types>(0))...,
    in_class_member_init(static_cast<Types>(0)) {} // expected-error{{initializer contains unexpanded parameter pack 'Types'}}

  void default_function_args(T = static_cast<Types>(0)); // expected-error{{default argument contains unexpanded parameter pack 'Types'}}

  template<typename = Types*> // expected-error{{default argument contains unexpanded parameter pack 'Types'}}
    struct default_template_args_1; 
  template<int = static_cast<Types>(0)> // expected-error{{default argument contains unexpanded parameter pack 'Types'}}
    struct default_template_args_2;
  template<template<typename> class = Types::template apply> // expected-error{{default argument contains unexpanded parameter pack 'Types'}}
    struct default_template_args_3;

  template<Types value> // expected-error{{non-type template parameter type contains unexpanded parameter pack 'Types'}}
  struct non_type_template_param_type;

  void decls_in_stmts() {
    Types t; // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    for (Types *t = 0; ; ) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    for (; Types *t = 0; ) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    T a[] = { T(), T(), T() };
    for (Types t : a) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    switch(Types *t = 0) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    while(Types *t = 0) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    if (Types *t = 0) { } // expected-error{{declaration type contains unexpanded parameter pack 'Types'}}
    try {
    } catch (Types*) { // expected-error{{exception type contains unexpanded parameter pack 'Types'}}
    }
  }
};

// FIXME: Test for unexpanded parameter packs in each of the statements.
struct X {
  void f(int, int);
  template<typename ...Types>
  void f(Types...);
};

namespace std {
  class type_info;
}

typedef struct _GUID {
     unsigned long  Data1;
     unsigned short Data2;
     unsigned short Data3;
     unsigned char  Data4[ 8 ];
} GUID;

template<typename T, typename ...Types>
void test_unexpanded_exprs(Types ...values) {
  // CXXOperatorCallExpr
  (void)(values + 0); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  (void)(0 + values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXMemberCallExpr
  values.f(); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  X x;
  x.f(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  x.Types::f(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  x.f<Types>(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // CXXStaticCastExpr
  (void)static_cast<Types&>(values); // expected-error{{expression contains unexpanded parameter packs 'Types' and 'values'}}

  // CXXDynamicCastExpr
  (void)dynamic_cast<Types&>(values); // expected-error{{expression contains unexpanded parameter packs 'Types' and 'values'}}

  // CXXReinterpretCastExpr
  (void)reinterpret_cast<Types&>(values); // expected-error{{expression contains unexpanded parameter packs 'Types' and 'values'}}

  // CXXConstCastExpr
  (void)const_cast<Types&>(values); // expected-error{{expression contains unexpanded parameter packs 'Types' and 'values'}}

  // CXXTypeidExpr
  (void)typeid(Types); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  (void)typeid(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXUuidofExpr
  (void)__uuidof(Types); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  (void)__uuidof(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXThisExpr is uninteresting

  // CXXThrowExpr
  throw Types(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  throw values; // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXDefaultArgExpr is uninteresting

  // CXXBindTemporaryExpr is uninteresting

  // CXXConstructExpr is uninteresting

  // CXXFunctionalCastExpr
  (void)Types(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // CXXTemporaryObjectExpr
  (void)X(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXScalarValueInitExpr is uninteresting

  // CXXNewExpr
  (void)new Types; // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  (void)new X(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  (void)new (values) X(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  (void)new X [values]; // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXDeleteExpr
  delete values; // expected-error{{expression contains unexpanded parameter pack 'values'}}
  delete [] values; // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXPseudoDestructorExpr
  T t;
  values.~T(); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  t.~Types(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  t.Types::~T(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // Unary TypeTraitExpr
  __is_pod(Types); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // Binary TypeTraitExpr
  __is_base_of(Types, T); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  __is_base_of(T, Types); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // UnresolvedLookupExpr
  test_unexpanded_exprs(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  test_unexpanded_exprs<Types>(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // DependentScopeDeclRefExpr
  Types::test_unexpanded_exprs(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  T::template test_unexpanded_exprs<Types>(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // CXXUnresolvedConstructExpr
  Types(5); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // CXXDependentScopeMemberExpr
  values.foo(); // expected-error{{expression contains unexpanded parameter pack 'values'}}
  t.foo(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // FIXME: There's an evil ambiguity here, because we don't know if
  // Types refers to the template type parameter pack in scope or a
  // non-pack member.
  //  t.Types::foo();

  t.template foo<Types>(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}

  // UnresolvedMemberExpr
  x.f<Types>(); // expected-error{{expression contains unexpanded parameter pack 'Types'}}
  x.f(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // CXXNoexceptExpr
  noexcept(values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

  // PackExpansionExpr is uninteresting
  // SizeOfPackExpr is uninteresting

  // FIXME: Objective-C expressions will need to go elsewhere

  for (auto t : values) { } // expected-error{{expression contains unexpanded parameter pack 'values'}}

  switch (values) { } // expected-error{{expression contains unexpanded parameter pack 'values'}}
  switch (0) { case 0: case values: ; } // expected-error{{expression contains unexpanded parameter pack 'values'}}

  do { } while (values); // expected-error{{expression contains unexpanded parameter pack 'values'}}

test:
  goto *values; // expected-error{{expression contains unexpanded parameter pack 'values'}}

  void f(int arg = values); // expected-error{{default argument contains unexpanded parameter pack 'values'}}
}

// Test unexpanded parameter packs in partial specializations.
template<typename ...Types>
struct TestUnexpandedDecls<int, Types>; // expected-error{{partial specialization contains unexpanded parameter pack 'Types'}}

// Test for diagnostics in the presence of multiple unexpanded
// parameter packs.
template<typename T, typename U> struct pair;

template<typename ...OuterTypes>
struct MemberTemplatePPNames {
  template<typename ...InnerTypes>
  struct Inner {
    typedef pair<OuterTypes, InnerTypes>* types; // expected-error{{declaration type contains unexpanded parameter packs 'OuterTypes' and 'InnerTypes'}}

    template<typename ...VeryInnerTypes>
    struct VeryInner {
      typedef pair<pair<VeryInnerTypes, OuterTypes>, pair<InnerTypes, OuterTypes> > types; // expected-error{{declaration type contains unexpanded parameter packs 'VeryInnerTypes', 'OuterTypes', ...}}
    };
  };
};

// Example from working paper
namespace WorkingPaperExample {
  template<typename...> struct Tuple {}; 
  template<typename T1, typename T2> struct Pair {};
  
  template<class ... Args1> struct zip { 
    template<class ... Args2> struct with {
      typedef Tuple<Pair<Args1, Args2> ... > type; // expected-error{{pack expansion contains parameter packs 'Args1' and 'Args2' that have different lengths (1 vs. 2)}}
    }; 
  };

  typedef zip<short, int>::with<unsigned short, unsigned>::type T1; // T1 is Tuple<Pair<short, unsigned short>, Pair<int, unsigned>>
  typedef Tuple<Pair<short, unsigned short>, Pair<int, unsigned>> T1;

  typedef zip<short>::with<unsigned short, unsigned>::type T2; // expected-note{{in instantiation of template class}}

  template<class ... Args> void f(Args...);
  template<class ... Args> void h(Args...);

  template<class ... Args> 
  void g(Args ... args) {
    f(const_cast<const Args*>(&args)...); // OK: "Args" and "args" are expanded within f 
    f(5 ...); // expected-error{{pack expansion does not contain any unexpanded parameter packs}}
    f(args); // expected-error{{expression contains unexpanded parameter pack 'args'}}
    f(h(args ...) + args ...);
  }
}

namespace PR16303 {
  template<int> struct A { A(int); };
  template<int...N> struct B {
    template<int...M> struct C : A<N>... {
      C() : A<N>(M)... {} // expected-error{{pack expansion contains parameter packs 'N' and 'M' that have different lengths (2 vs. 3)}} expected-error{{pack expansion contains parameter packs 'N' and 'M' that have different lengths (4 vs. 3)}}
    };
  };
  B<1,2>::C<4,5,6> c1; // expected-note{{in instantiation of}}
  B<1,2,3,4>::C<4,5,6> c2; // expected-note{{in instantiation of}}
}

namespace PR21289 {
  template<int> using T = int;
  template<typename> struct S { static const int value = 0; };
  template<typename> const int vt = 0; // cxx11-warning {{extension}}
  int f(...);
  template<int ...Ns> void g() {
    f(T<Ns>()...);
    f(S<T<Ns>>::value...);
    f(vt<T<Ns>>...);
  }
  template void g<>();
  template void g<1, 2, 3>();
}

template <class... Ts>
int var_expr(Ts... ts);

template <class... Ts>
auto a_function(Ts... ts) -> decltype(var_expr(ts...));

template <class T>
using partial = decltype(a_function<int, T>);

int use_partial() { partial<char> n; }

namespace PR26017 {
template <class T>
struct Foo {};
template <class... Ts>
using FooAlias = Foo<void(Ts...)>;

template <class... Ts>
using FooAliasAlias = FooAlias<Ts..., Ts...>;

template <class... Ts>
void bar(const FooAlias<Ts...> &) {}

int fn() {
  FooAlias<> a;
  bar(a);

  FooAlias<int> b;
  bar(b);
}
}

namespace GH58452 {
template <typename... As> struct A {
  template <typename... Bs> using B = void(As...(Bs));
};

template <typename... Cs> struct C {
    template <typename... Ds> using D = typename A<Cs...>::template B<Ds...>;
};

using t1 = C<int, int>::template D<float, float>;

template <typename A, typename B>
using ConditionalRewrite = B;

template <typename T>
using SignatureType = int;

template <typename... Args>
struct Type1 {
    template <typename... Params>
        using Return = SignatureType<int(ConditionalRewrite<Args, Params>...)>;

};

template <typename... Args>
struct Type2 {
    using T1 = Type1<Args...>;

      template <typename... Params>
          using Return = typename T1::template Return<Params...>;

};

template <typename T>
typename T::template Return<int, int> InvokeMethod() {
    return 3;
}

int Function1() {
    return InvokeMethod<Type2<int, int>>();
}
}
