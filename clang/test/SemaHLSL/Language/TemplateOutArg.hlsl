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

// Case 4: Verify that the parameter modifier attributes are instantiated 
// for both templated and non-templated arguments, and that the non-templated
// out argument type is not modified by the template instantiation.

// CHECK-LABEL: FunctionTemplateDecl {{.*}} fizz_two

// Check the pattern decl.
// CHECK: FunctionDecl {{.*}} fizz_two 'void (inout T, out int)'
// CHECK-NEXT: ParmVarDecl {{.*}} referenced V 'T'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK-NEXT: ParmVarDecl {{.*}} referenced I 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out

// Check the 3 instantiations (int, float, & double).

// CHECK-LABEL: FunctionDecl {{.*}} used fizz_two 'void (inout int, out int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK: ParmVarDecl {{.*}} used I 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out

// CHECK-LABEL: FunctionDecl {{.*}} used fizz_two 'void (inout float, out int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK: ParmVarDecl {{.*}} used I 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out

// CHECK-LABEL: FunctionDecl {{.*}} used fizz_two 'void (inout double, out int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used V 'double &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK: ParmVarDecl {{.*}} used I 'int &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
template <typename T>
void fizz_two(inout T V, out int I) {
  V += 2;
  I = V;
}

// Case 5: Verify that `in` parameter modifier attributes are instantiated 
// for both templated and non-templated arguments and argument types are not
// modified

// CHECK-LABEL: FunctionTemplateDecl {{.*}} buzz_two

// Check the pattern decl.
// CHECK: FunctionDecl {{.*}} buzz_two 'int (T, int)'
// CHECK-NEXT: ParmVarDecl {{.*}} referenced A 'T'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
// CHECK-NEXT: ParmVarDecl {{.*}} referenced B 'int'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in

// Check the 3 instantiations (int, float, & double).

// CHECK-LABEL: FunctionDecl {{.*}} used buzz_two 'int (int, int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used A 'int'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
// CHECK: ParmVarDecl {{.*}} used B 'int'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in

// CHECK-LABEL: FunctionDecl {{.*}} used buzz_two 'int (float, int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used A 'float'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
// CHECK: ParmVarDecl {{.*}} used B 'int'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in

// CHECK-LABEL: FunctionDecl {{.*}} used buzz_two 'int (double, int)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} used A 'double'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
// CHECK: ParmVarDecl {{.*}} used B 'int'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
template <typename T>
int buzz_two(in T A, in int B) {
  return A + B;
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

  fizz_two(X, X);
  fizz_two(Y, X);
  fizz_two(Z, X);

  X = buzz_two(X, X);
  X = buzz_two(Y, X);
  X = buzz_two(Z, X);
}
