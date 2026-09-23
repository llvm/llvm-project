// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

struct X { int a; };

using as1_X = X __attribute__((address_space(1)));

as1_X prvalue();
as1_X &lvalue();
as1_X &&xvalue();

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
};

using as1_int = __attribute__((address_space(1))) int;
using const_as1_int = const __attribute__((address_space(1))) int;
using as1_S = __attribute__((address_space(1))) S;
using const_as1_S = const __attribute__((address_space(1))) S;

void qualifiersTest() {
  // The address space and const qualifier are dropped on a prvalue of non-class
  // type.
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to as1_int
  (void)(as1_int{1});
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to const_as1_int
  (void)(const_as1_int{1});
  // CHECK: CXXFunctionalCastExpr {{.*}} 'int' functional cast to const_as1_int <NoOp>
  (void)const_as1_int(0);
  // CHECK: CStyleCastExpr {{.*}} 'int' <NoOp>
  (void)(const_as1_int)0;
  // CHECK: CXXStaticCastExpr {{.*}} 'int' static_cast<const_as1_int> <NoOp>
  (void)static_cast<const_as1_int>(0);

  // The address space is dropped on a prvalue of class type
  // const is retained.
  // CHECK: CXXStaticCastExpr {{.*}} 'S' static_cast<as1_S> <NoOp>
  (void)static_cast<as1_S>(S{0});
  // CHECK: CXXStaticCastExpr {{.*}} 'const S' static_cast<const_as1_S> <NoOp>
  (void)static_cast<const_as1_S>(S{0});
}

as1_int &&rvalueRefInt();
as1_X &&rvalueRefX();

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

