// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -finclude-default-header %s -ast-dump | FileCheck %s

// Case 1: Template declaration with a call to an inout or out argument that is
// resolved based on the template parameter. For this case the template decl
// should have an UnresolvedLookupExpr for the call, and the HLSLOutArgExpr is
// built during call resolution.

// CHECK: FunctionDecl {{.*}} used fn 'void (inout int)'
void fn(inout int I) {
  I += 1;
}

// CHECK: FunctionDecl {{.*}} used fn 'void (out double)'
void fn(out double F) {
  F = 1.5;
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} wrapper
// CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T

// Verify that the template has an unresolved call.
// CHECK-NEXT: FunctionDecl {{.*}} wrapper 'T (T)'
// CHECK-NEXT: ParmVarDecl {{.*}} referenced V 'T'
// CHECK: CallExpr {{.*}} '<dependent type>'
// CHECK: UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'fn'

// Verify that the int instantiation resolves an inout argument expression.

// CHECK-LABEL: FunctionDecl {{.*}} used wrapper 'int (int)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}} 'void (inout int)' lvalue Function {{.*}} 'fn' 'void (inout int)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout

// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'V' 'int'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue

// CHECK: BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue


// Verify that the float instantiation has an out argument expression
// containing casts to and from double.

// CHECK-LABEL: FunctionDecl {{.*}} used wrapper 'float (float)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(out double)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}}'void (out double)' lvalue Function {{.*}} 'fn' 'void (out double)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'double' lvalue out
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'float' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'V' 'float'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'double' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'float' lvalue

// CHECK: BinaryOperator {{.*}} 'float' lvalue '='
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'float' lvalue
// CHECK: ImplicitCastExpr {{.*}} 'float' <FloatingCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV]] {{.*}} 'double' lvalue


// Verify that the double instantiation is just an out expression.

// CHECK-LABEL: FunctionDecl {{.*}} used wrapper 'double (double)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(out double)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}}'void (out double)' lvalue Function {{.*}} 'fn' 'void (out double)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'double' lvalue out
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'double' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'double' lvalue ParmVar {{.*}} 'V' 'double'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'double' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'double' lvalue

// CHECK: BinaryOperator {{.*}} 'double' lvalue '='
// CHECK-NEXT: OpaqueValueExpr [[LVOpV]] {{.*}} 'double' lvalue
// CHECK: ImplicitCastExpr {{.*}} 'double' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV]] {{.*}} 'double' lvalue

template <typename T>
T wrapper(T V) {
  fn(V);
  return V;
}

// Case 2: Verify that the parameter modifier attribute is instantiated with the
// template (this one is a gimme).

// CHECK-LABEL: FunctionTemplateDecl {{.*}} fizz

// Check the pattern decl.
// CHECK: FunctionDecl {{.*}} fizz 'void (inout T)'
// CHECK-NEXT: ParmVarDecl {{.*}} referenced V 'T'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout

// Check the 3 instantiations (int, float, & double).

// CHECK-LABEL: FunctionDecl {{.*}} used fizz 'void (inout int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout

// CHECK-LABEL: FunctionDecl {{.*}} used fizz 'void (inout float)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout

// CHECK-LABEL: FunctionDecl {{.*}} used fizz 'void (inout double)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'double &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
template <typename T>
void fizz(inout T V) {
  V += 2;
}

// Case 3: Verify that HLSLOutArgExpr nodes which are present in the template
// are correctly instantiated into the instantation.

// First we check that the AST node is in the template.

// CHECK-LABEL: FunctionTemplateDecl {{.*}} buzz

// CHECK: FunctionDecl {{.*}} buzz 'T (int, T)'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}} 'void (inout int)' lvalue Function {{.*}} 'fn' 'void (inout int)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   DeclRefExpr  {{.*}} 'int' lvalue ParmVar {{.*}} 'X' 'int'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:     OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT:   OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue



// CHECK-LABEL: FunctionDecl {{.*}} used buzz 'int (int, int)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}} 'void (inout int)' lvalue Function {{.*}} 'fn' 'void (inout int)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   DeclRefExpr  {{.*}} 'int' lvalue ParmVar {{.*}} 'X' 'int'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:     OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT:   OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue


// CHECK-LABEL: FunctionDecl {{.*}} used buzz 'float (int, float)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}} 'void (inout int)' lvalue Function {{.*}} 'fn' 'void (inout int)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   DeclRefExpr  {{.*}} 'int' lvalue ParmVar {{.*}} 'X' 'int'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:     OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT:   OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue


// CHECK-LABEL: FunctionDecl {{.*}} used buzz 'double (int, double)' implicit_instantiation
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout int)' <FunctionToPointerDecay>
// CHECK-NEXT:   DeclRefExpr {{.*}} 'void (inout int)' lvalue Function {{.*}} 'fn' 'void (inout int)'
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'int' lvalue inout
// CHECK-NEXT: OpaqueValueExpr [[LVOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   DeclRefExpr  {{.*}} 'int' lvalue ParmVar {{.*}} 'X' 'int'
// CHECK-NEXT: OpaqueValueExpr [[TmpOpV:0x[0-9a-fA-F]+]] {{.*}} 'int' lvalue
// CHECK-NEXT:   ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:     OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NEXT:   OpaqueValueExpr [[LVOpV]] {{.*}} 'int' lvalue
// CHECK:      ImplicitCastExpr  {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT:   OpaqueValueExpr [[TmpOpV]] {{.*}} 'int' lvalue

template <typename T>
T buzz(int X, T Y) {
  fn(X);
  return X + Y;
}

export void caller() {
  int X = 2;
  float Y = 3.3;
  double Z = 2.2;

  X = wrapper(X);
  Y = wrapper(Y);
  Z = wrapper(Z);

  fizz(X);
  fizz(Y);
  fizz(Z);

  X = buzz(X, X);
  Y = buzz(X, Y);
  Z = buzz(X, Z);
}
