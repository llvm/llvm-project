// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

struct X { int a; };

using GlobalX = X __attribute__((address_space(1)));

GlobalX prvalue();
GlobalX &lvalue();
GlobalX &&xvalue();

void test() {
  // A prvalue should not have an address space even if the function's
  // return type is address-space qualified.
  // CHECK: VarDecl {{.*}} v 'X'
  // CHECK: CallExpr {{.*}} 'X'{{$}}
  auto v = prvalue();

  // CHECK: VarDecl {{.*}} l '__attribute__((address_space(1))) X &'
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' lvalue
  auto &l = lvalue();

  // CHECK: VarDecl {{.*}} r '__attribute__((address_space(1))) X &&'
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' xvalue
  auto &&r = xvalue();
}

struct S {
  S(int);
  void f() const;
};

using const_int = const int;
using as1_int = __attribute__((address_space(1))) int;
using const_S = const S;

void qualifierTest() {
  // const is retained on prvalues of class type; const qualified
  // member function called.
  // CHECK: MaterializeTemporaryExpr {{.*}} 'const_S':'const S' xvalue
  // CHECK: CXXTemporaryObjectExpr {{.*}} 'const_S':'const S'
  const_S{1}.f();

  // const is dropped on prvalues of non-class type.
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to const_int
  (void)(const_int{1});

  // address space is dropped on prvalues of non-class type.
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to as1_int
  (void)(as1_int{1});
}

void castTest() {
  // The address space is dropped on the prvalue produced by a cast to a
  // non-class type.
  // CHECK: CStyleCastExpr {{.*}} 'int' <NoOp>
  (void)(as1_int)0;
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to as1_int <NoOp>
  (void)as1_int(0);
  // CHECK: CXXStaticCastExpr {{.*}} 'int' static_cast<as1_int> <NoOp>
  (void)static_cast<as1_int>(0);
}

as1_int &&rvalueRefInt();
GlobalX &&rvalueRefX();

void rvalueReferenceTest() {
  // An rvalue reference is an xvalue naming an object, not a prvalue, so the
  // address space is kept.
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) int' xvalue
  (void)rvalueRefInt();
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' xvalue
  (void)rvalueRefX();
}

void vaArgTest(__builtin_va_list va) {
  // A __builtin_va_arg expression is a prvalue. Both const and the address
  // space are dropped for non-class types.
  // CHECK: VAArgExpr {{.*}} 'int'
  __builtin_va_arg(va, const __attribute__((address_space(1))) int);

  // const is retained but the address space is dropped for class types.
  // CHECK: VAArgExpr {{.*}} 'const X'
  __builtin_va_arg(va, const __attribute__((address_space(1))) X);
}

// A non-type template parameter of address-space-qualified type is a prvalue
// of the unqualified type.
// CHECK: NonTypeTemplateParmDecl {{.*}} 'int' {{.*}} N
template <as1_int N> struct NT {};
NT<0> nt;

