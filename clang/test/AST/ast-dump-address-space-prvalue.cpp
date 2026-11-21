// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

struct X {
  void f() const;
};

using as1_int = __attribute__((address_space(1))) int;
using const_as1_int = const __attribute__((address_space(1))) int;
using as1_X = X __attribute__((address_space(1)));
using const_as1_X = const __attribute__((address_space(1))) X;

as1_X prvalue();
as1_int prvalueInt();
const_as1_X constPrvalue();
const_as1_int constPrvalueInt();
as1_X &lvalue();
as1_int &lvalueInt();
as1_X &&xvalue();
as1_int &&xvalueInt();

void callResult() {
  // A prvalue should not have an address space even if the function's
  // return type is address-space qualified.
  // CHECK: CallExpr {{.*}} 'X'{{$}}
  (void)prvalue();
  // CHECK: CallExpr {{.*}} 'int'{{$}}
  (void)prvalueInt();

  // const is retained on a class prvalue.
  // CHECK: CallExpr {{.*}} 'const X'{{$}}
  (void)constPrvalue();
  // CHECK: CallExpr {{.*}} 'int'{{$}}
  (void)constPrvalueInt();

  // A reference names an object in the address space.
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' lvalue
  (void)lvalue();
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) int' lvalue
  (void)lvalueInt();
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' xvalue
  (void)xvalue();
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) int' xvalue
  (void)xvalueInt();
}

struct M {
  as1_X m();
  as1_X operator-();
  as1_X operator+(int);
  as1_X operator[](int);
  as1_X operator()();
  operator as1_X();
};

struct MInt {
  as1_int m();
  as1_int operator-();
  as1_int operator+(int);
  as1_int operator[](int);
  as1_int operator()();
  operator as1_int();
};

// Calls to overloaded operators, member functions and conversion
// functions also produce a prvalue.
void overloadedCalls(M mm, MInt mi) {
  // CHECK: CXXMemberCallExpr {{.*}} 'X'{{$}}
  mm.m().f();
  // CHECK: CXXMemberCallExpr {{.*}} 'int'{{$}}
  (void)mi.m();
  // CHECK: CXXOperatorCallExpr {{.*}} 'X' '-'
  (-mm).f();
  // CHECK: CXXOperatorCallExpr {{.*}} 'int' '-'
  (void)-mi;
  // CHECK: CXXOperatorCallExpr {{.*}} 'X' '+'
  (mm + 1).f();
  // CHECK: CXXOperatorCallExpr {{.*}} 'int' '+'
  (void)(mi + 1);
  // CHECK: CXXOperatorCallExpr {{.*}} 'X' '[]'
  mm[0].f();
  // CHECK: CXXOperatorCallExpr {{.*}} 'int' '[]'
  (void)mi[0];
  // CHECK: CXXOperatorCallExpr {{.*}} 'X' '()'
  mm().f();
  // CHECK: CXXOperatorCallExpr {{.*}} 'int' '()'
  (void)mi();
  // CHECK: CXXMemberCallExpr {{.*}} 'X'{{$}}
  X c = mm;
  // CHECK: CXXMemberCallExpr {{.*}} 'int'{{$}}
  int i = mi;
}

// A by-value return initializes a result object without the address space.
// The declared function type keeps the address space.
// CHECK: FunctionDecl {{.*}} fnReturn 'as1_X ()'
// CHECK: ReturnStmt
// CHECK-NEXT: CXXConstructExpr {{.*}} 'X' 'void (const X &) noexcept'
X x;
as1_X fnReturn() { return x; }
// CHECK: FunctionDecl {{.*}} fnReturnInt 'as1_int ()'
// CHECK: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
int i;
as1_int fnReturnInt() { return i; }

// A cast to a non-class type ignores top-level qualifiers, so the address
// space and const are dropped and the result is a plain value. A class type
// keeps its qualifiers, so a cast or temporary of an address-space-qualified
// class type is rejected (see SemaCXX/address-space-prvalue.cpp).
void castTest() {
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
