// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -ast-dump %s | FileCheck %s

// We are checking that implicit casts don't get marked with 'part_of_explicit_cast',
// while in explicit casts, the implicitly-inserted implicit casts are marked with 'part_of_explicit_cast'

unsigned char implicitcast_0(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} implicitcast_0 'unsigned char (unsigned int)'{{$}}
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue>{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return x;
}

signed char implicitcast_1(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} implicitcast_1 'signed char (unsigned int)'{{$}}
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue>{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return x;
}

unsigned char implicitcast_2(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} implicitcast_2 'unsigned char (int)'{{$}}
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue>{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return x;
}

signed char implicitcast_3(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} implicitcast_3 'signed char (int)'{{$}}
  // CHECK: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue>{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return x;
}

//----------------------------------------------------------------------------//

unsigned char cstylecast_0(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_0 'unsigned char (unsigned int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return (unsigned char)x;
}

signed char cstylecast_1(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_1 'signed char (unsigned int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'signed char' <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return (signed char)x;
}

unsigned char cstylecast_2(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_2 'unsigned char (int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return (unsigned char)x;
}

signed char cstylecast_3(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_3 'signed char (int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'signed char' <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return (signed char)x;
}

//----------------------------------------------------------------------------//

unsigned char cxxstaticcast_0(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxstaticcast_0 'unsigned char (unsigned int)'{{$}}
  // CHECK: CXXStaticCastExpr {{.*}} <col:{{.*}}> 'unsigned char' static_cast<unsigned char> <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return static_cast<unsigned char>(x);
}

signed char cxxstaticcast_1(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxstaticcast_1 'signed char (unsigned int)'{{$}}
  // CHECK: CXXStaticCastExpr {{.*}} <col:{{.*}}> 'signed char' static_cast<signed char> <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return static_cast<signed char>(x);
}

unsigned char cxxstaticcast_2(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxstaticcast_2 'unsigned char (int)'{{$}}
  // CHECK: CXXStaticCastExpr {{.*}} <col:{{.*}}> 'unsigned char' static_cast<unsigned char> <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return static_cast<unsigned char>(x);
}

signed char cxxstaticcast_3(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxstaticcast_3 'signed char (int)'{{$}}
  // CHECK: CXXStaticCastExpr {{.*}} <col:{{.*}}> 'signed char' static_cast<signed char> <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return static_cast<signed char>(x);
}

//----------------------------------------------------------------------------//

using UnsignedChar = unsigned char;
using SignedChar = signed char;
using UnsignedInt = unsigned int;
using SignedInt = signed int;

UnsignedChar cxxfunctionalcast_0(UnsignedInt x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxfunctionalcast_0 'UnsignedChar (UnsignedInt)'{{$}}
  // CHECK: CXXFunctionalCastExpr {{.*}} <col:{{.*}}> 'UnsignedChar':'unsigned char' functional cast to UnsignedChar <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'UnsignedChar':'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'UnsignedInt':'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'UnsignedInt':'unsigned int' lvalue ParmVar {{.*}} 'x' 'UnsignedInt':'unsigned int'{{$}}
  return UnsignedChar(x);
}

SignedChar cxxfunctionalcast_1(UnsignedInt x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxfunctionalcast_1 'SignedChar (UnsignedInt)'{{$}}
  // CHECK: CXXFunctionalCastExpr {{.*}} <col:{{.*}}> 'SignedChar':'signed char' functional cast to SignedChar <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'SignedChar':'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'UnsignedInt':'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'UnsignedInt':'unsigned int' lvalue ParmVar {{.*}} 'x' 'UnsignedInt':'unsigned int'{{$}}
  return SignedChar(x);
}

UnsignedChar cxxfunctionalcast_2(SignedInt x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxfunctionalcast_2 'UnsignedChar (SignedInt)'{{$}}
  // CHECK: CXXFunctionalCastExpr {{.*}} <col:{{.*}}> 'UnsignedChar':'unsigned char' functional cast to UnsignedChar <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'UnsignedChar':'unsigned char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'SignedInt':'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'SignedInt':'int' lvalue ParmVar {{.*}} 'x' 'SignedInt':'int'{{$}}
  return UnsignedChar(x);
}

SignedChar cxxfunctionalcast_3(SignedInt x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cxxfunctionalcast_3 'SignedChar (SignedInt)'{{$}}
  // CHECK: CXXFunctionalCastExpr {{.*}} <col:{{.*}}> 'SignedChar':'signed char' functional cast to SignedChar <NoOp>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'SignedChar':'signed char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'SignedInt':'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'SignedInt':'int' lvalue ParmVar {{.*}} 'x' 'SignedInt':'int'{{$}}
  return SignedChar(x);
}
