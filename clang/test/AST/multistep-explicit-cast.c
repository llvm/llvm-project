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
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return (unsigned char)x;
}

signed char cstylecast_1(unsigned int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_1 'signed char (unsigned int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'unsigned int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'unsigned int' lvalue ParmVar {{.*}} 'x' 'unsigned int'{{$}}
  return (signed char)x;
}

unsigned char cstylecast_2(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_2 'unsigned char (int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'unsigned char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return (unsigned char)x;
}

signed char cstylecast_3(signed int x) {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} cstylecast_3 'signed char (int)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'signed char' <IntegralCast>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'int' <LValueToRValue> part_of_explicit_cast{{$}}
  // CHECK-NEXT: DeclRefExpr {{.*}} <col:{{.*}}> 'int' lvalue ParmVar {{.*}} 'x' 'int'{{$}}
  return (signed char)x;
}
