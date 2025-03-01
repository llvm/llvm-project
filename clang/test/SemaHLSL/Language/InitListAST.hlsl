// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -finclude-default-header -ast-dump -ast-dump-filter=case %s | FileCheck %s

struct TwoFloats {
  float X, Y;
};

struct TwoInts {
  int Z, W;
};

struct Doggo {
  int4 LegState;
  int TailState;
  float HairCount;
  float4 EarDirection[2];
};

struct AnimalBits {
  int Legs[4];
  uint State;
  int64_t Counter;
  float4 LeftDir;
  float4 RightDir;
};

struct Kitteh {
  int4 Legs;
  int TailState;
  float HairCount;
  float4 Claws[2];
};

struct Zoo {
  Doggo Dogs[2];
  Kitteh Cats[4];
};

struct FourFloats : TwoFloats {
  float Z, W;
};

struct SlicyBits {
  int Z : 8;
  int W : 8;
};

// Case 1: Extraneous braces get ignored in literal instantiation.
// CHECK-LABEL: Dumping case1
// CHECK: VarDecl {{.*}} used TF1 'TwoFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
TwoFloats case1() {
  TwoFloats TF1 = {{{1.0, 2}}};
  return TF1;
}

// Case 2: Valid C/C++ initializer is handled appropriately.
//CHECK-LABEL: Dumping case2
//CHECK: VarDecl {{.*}} used TF2 'TwoFloats' nrvo cinit
//CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
//CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
//CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
TwoFloats case2() {
  TwoFloats TF2 = {1, 2};
  return TF2;
}

// Case 3: Simple initialization with conversion of an argument.
// CHECK-LABEL: Dumping case3
// CHECK: VarDecl {{.*}} used TF3 'TwoFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'Val' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
TwoFloats case3(int Val) {
  TwoFloats TF3 = {Val, 2};
  return TF3;
}

// Case 4: Initialization from a scalarized vector into a structure with element
// conversions.
// CHECK-LABEL: Dumping case4
// CHECK: VarDecl {{.*}} used TF4 'TwoFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}}'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'TwoVals' 'int2':'vector<int, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'TwoVals' 'int2':'vector<int, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
TwoFloats case4(int2 TwoVals) {
  TwoFloats TF4 = {TwoVals};
  return TF4;
}

// Case 5: Initialization from a scalarized vector of matching type.
// CHECK-LABEL: Dumping case5
// CHECK: VarDecl {{.*}} used TI1 'TwoInts' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'TwoVals' 'int2':'vector<int, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'int2':'vector<int, 2>' lvalue ParmVar {{.*}} 'TwoVals' 'int2':'vector<int, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
TwoInts case5(int2 TwoVals) {
  TwoInts TI1 = {TwoVals};
  return TI1;
}

// Case 6: Initialization from a scalarized structure of different type with
// different element types.
// CHECK-LABEL: Dumping case6
// CHECK: VarDecl {{.*}} used TI2 'TwoInts' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}}'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF4' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF4' 'TwoFloats'
TwoInts case6(TwoFloats TF4) {
  TwoInts TI2 = {TF4};
  return TI2;
}

// Case 7: Initialization of a complex structure, with bogus braces and element
// conversions from a collection of scalar values, and structures.
// CHECK-LABEL: Dumping case7
// CHECK: VarDecl {{.*}} used D1 'Doggo' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .Z {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI1' 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .W {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI1' 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .Z {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI2' 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .W {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI2' 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'Val' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'Val' 'int'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF1' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF1' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF2' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF2' 'TwoFloats'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF3' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF3' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF4' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF4' 'TwoFloats'
Doggo case7(TwoInts TI1, TwoInts TI2, int Val, TwoFloats TF1, TwoFloats TF2,
            TwoFloats TF3, TwoFloats TF4) {
  Doggo D1 = {TI1, TI2, {Val, Val}, {{TF1, TF2}, {TF3, TF4}}};
  return D1;
}

// Case 8: Initialization of a structure from a different structure with
// significantly different element types and grouping.
// CHECK-LABEL: Dumping case8
// CHECK: VarDecl {{.*}} used A1 'AnimalBits' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'AnimalBits'
// CHECK-NEXT: InitListExpr {{.*}} 'int[4]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .TailState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'long' <FloatingToIntegral>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .HairCount {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
AnimalBits case8(Doggo D1) {
  AnimalBits A1 = {D1};
  return A1;
}

// Case 9: Everything everywhere all at once... Initializing mismatched
// structures from different layouts, different component groupings, with no
// top-level bracing separation.
// CHECK-LABEL: Dumping case9
// CHECK: VarDecl {{.*}} used Z1 'Zoo' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'Zoo'
// CHECK-NEXT: InitListExpr {{.*}} 'Doggo[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .TailState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .HairCount {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint':'unsigned int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'uint':'unsigned int' lvalue .State {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t':'long' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int64_t':'long' lvalue .Counter {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'Kitteh[4]'
// CHECK-NEXT: InitListExpr {{.*}} 'Kitteh'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .TailState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .HairCount {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'Kitteh'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint':'unsigned int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'uint':'unsigned int' lvalue .State {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t':'long' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int64_t':'long' lvalue .Counter {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'Kitteh'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'int4':'vector<int, 4>' lvalue .LegState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .TailState {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .HairCount {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float4':'vector<float, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float4 *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'float4[2]' lvalue .EarDirection {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'Doggo' lvalue ParmVar {{.*}} 'D1' 'Doggo'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'Kitteh'
// CHECK-NEXT: InitListExpr {{.*}} 'int4':'vector<int, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'int' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'int[4]' lvalue .Legs {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'uint':'unsigned int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'uint':'unsigned int' lvalue .State {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int64_t':'long' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int64_t':'long' lvalue .Counter {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: InitListExpr {{.*}} 'float4[2]'
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .LeftDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: MemberExpr {{.*}} 'float4':'vector<float, 4>' lvalue .RightDir {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'AnimalBits' lvalue ParmVar {{.*}} 'A1' 'AnimalBits'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
Zoo case9(Doggo D1, AnimalBits A1) {
  Zoo Z1 = {D1, A1, D1, A1, D1, A1};
  return Z1;
}

// Case 10: Initialize an object with a base class from two objects.
// CHECK-LABEL: Dumping case10
// CHECK: | `-VarDecl {{.*}} used FF1 'FourFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'FourFloats'
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF1' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF1' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .X {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF2' 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .Y {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoFloats' lvalue ParmVar {{.*}} 'TF2' 'TwoFloats'
FourFloats case10(TwoFloats TF1, TwoFloats TF2) {
  FourFloats FF1 = {TF1, TF2};
  return FF1;
}

// Case 11: Initialize an object with a base class from a vector splat.
// CHECK-LABEL: Dumping case11
// CHECK: VarDecl {{.*}} used FF1 'FourFloats' nrvo cinit
// CHECK-NEXT: ExprWithCleanups {{.*}} 'FourFloats'
// CHECK-NEXT: InitListExpr {{.*}} 'FourFloats'
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'vector<float, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'F' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'vector<float, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'F' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'vector<float, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'F' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' xvalue vectorcomponent
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'vector<float, 4>' xvalue
// CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 4>' xxxx
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 1>' lvalue <VectorSplat>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} 'F' 'float'
// CHECK-NEXT: IntegerLiteral {{.*}} 'unsigned long' 3
FourFloats case11(float F) {
  FourFloats FF1 = {F.xxxx};
  return FF1;
}

// Case 12: Initialize bitfield from two integers.
// CHECK-LABEL: Dumping case12
// CHECK: VarDecl {{.*}} used SB 'SlicyBits' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'SlicyBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'I' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'J' 'int'
SlicyBits case12(int I, int J) {
  SlicyBits SB = {I, J};
  return SB;
}

// Case 13: Initialize bitfield from a struct of two ints.
// CHECK-LABEL: Dumping case13
// CHECK: VarDecl {{.*}} used SB 'SlicyBits' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'SlicyBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .Z {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI' 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue .W {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'TwoInts' lvalue ParmVar {{.*}} 'TI' 'TwoInts'
SlicyBits case13(TwoInts TI) {
  SlicyBits SB = {TI};
  return SB;
}

// Case 14: Initialize struct of ints from struct with bitfields.
// CHECK-LABEL: Dumping case14
// CHECK: VarDecl {{.*}} used TI 'TwoInts' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoInts'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue bitfield .Z {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'SlicyBits' lvalue ParmVar {{.*}} 'SB' 'SlicyBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue bitfield .W {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'SlicyBits' lvalue ParmVar {{.*}} 'SB' 'SlicyBits'
TwoInts case14(SlicyBits SB) {
  TwoInts TI = {SB};
  return TI;
}

// Case 15: Initialize struct of floats from struct with bitfields.
// CHECK-LABEL: Dumping case15
// CHECK: VarDecl {{.*}} used TI 'TwoFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue bitfield .Z {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'SlicyBits' lvalue ParmVar {{.*}} 'SB' 'SlicyBits'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'int' lvalue bitfield .W {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'SlicyBits' lvalue ParmVar {{.*}} 'SB' 'SlicyBits'
TwoFloats case15(SlicyBits SB) {
  TwoFloats TI = {SB};
  return TI;
}

// Case 16: Side-effecting initialization list arguments. The important thing
// here is that case16 only has _one_ call to makeTwo.
TwoFloats makeTwo(inout float X) {
    TwoFloats TF = {X, X*1.5};
    X *= 2;
    return TF;
}

// CHECK-LABEL: Dumping case16
// CHECK: VarDecl {{.*}} used FF 'FourFloats' nrvo cinit
// CHECK-NEXT: InitListExpr {{.*}} 'FourFloats'
// CHECK-NEXT: InitListExpr {{.*}} 'TwoFloats'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' xvalue .X {{.*}}
// CHECK-NEXT: OpaqueValueExpr [[OVEArg:0x[0-9A-Fa-f]+]] {{.*}} 'TwoFloats' xvalue
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'TwoFloats' xvalue
// CHECK-NEXT: CallExpr {{.*}} 'TwoFloats'

// I don't care about the call here, just skip ahead to the next argument, and
// verify that we match the same OpaqueValueExpr.

// CHECK: MemberExpr {{.*}} 'float' xvalue .Y {{.*}}
// CHECK-NEXT: OpaqueValueExpr [[OVEArg]] {{.*}} 'TwoFloats' xvalue
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'TwoFloats' xvalue
// CHECK-NEXT: CallExpr {{.*}} 'TwoFloats'
FourFloats case16() {
    float X = 0;
    FourFloats FF = {0, makeTwo(X), 3};
    return FF;
}
