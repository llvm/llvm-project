// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

int *Global;
// CHECK: VarDecl{{.*}}Global 'int *'
int GlobalArray[5];
// CHECK-NEXT: VarDecl{{.*}}GlobalArray 'int[5]'
#pragma acc declare deviceptr(Global), copyin(readonly, always: GlobalArray)
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: deviceptr clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int *'
// CHECK-NEXT: copyin clause modifiers: always, readonly
// CHECK-NEXT: DeclRefExpr{{.*}}'GlobalArray' 'int[5]'

int *Global1;
// CHECK-NEXT: VarDecl{{.*}}Global1 'int *'
int GlobalArray1[5];
// CHECK-NEXT: VarDecl{{.*}}GlobalArray1 'int[5]'
_Pragma("acc declare deviceptr(Global1), copyin(GlobalArray1)")
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: deviceptr clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Global1' 'int *'
// CHECK-NEXT: copyin clause
// CHECK-NEXT: DeclRefExpr{{.*}}'GlobalArray1' 'int[5]'

int *Global2;
// CHECK: VarDecl{{.*}}Global2 'int *'
int GlobalArray2[5];
// CHECK-NEXT: VarDecl{{.*}}GlobalArray2 'int[5]'
#pragma acc declare create(Global2, GlobalArray2)
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: create clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Global2' 'int *'
// CHECK-NEXT: DeclRefExpr{{.*}}'GlobalArray2' 'int[5]'

int Global3;
// CHECK: VarDecl{{.*}}Global3 'int'
int GlobalArray3[5];
// CHECK-NEXT: VarDecl{{.*}}GlobalArray3 'int[5]'
#pragma acc declare link(Global3) device_resident(GlobalArray3)
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: link clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Global3' 'int'
// CHECK-NEXT: device_resident clause
// CHECK-NEXT: DeclRefExpr{{.*}}'GlobalArray3' 'int[5]'

namespace NS {
int NSVar;
// CHECK: VarDecl{{.*}}NSVar 'int'
int NSArray[5];
// CHECK-NEXT: VarDecl{{.*}}NSArray 'int[5]'
#pragma acc declare create(NSVar, NSArray)
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: create clause
// CHECK-NEXT: DeclRefExpr{{.*}}'NSVar' 'int'
// CHECK-NEXT: DeclRefExpr{{.*}}'NSArray' 'int[5]'
}

struct Struct {
  // CHECK-NEXT: CXXRecordDecl{{.*}} Struct definition
  // Skip DefinitionData and go right to the definition.
  // CHECK: CXXRecordDecl{{.*}} implicit struct Struct
  static const int StaticMem = 5;
  // CHECK-NEXT: VarDecl {{.*}} StaticMem 'const int' static cinit
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 5
  static const int StaticMem2 = 5;
  // CHECK-NEXT: VarDecl {{.*}} StaticMem2 'const int' static cinit
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 5
  static const int StaticMemArray[5];
  // CHECK-NEXT: VarDecl {{.*}} StaticMemArray 'const int[5]' static
  static const int StaticMemArray2[5];
  // CHECK-NEXT: VarDecl {{.*}} StaticMemArray2 'const int[5]' static
#pragma acc declare copyin(StaticMem, StaticMemArray) create(StaticMem2, StaticMemArray2)
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem' 'const int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray' 'const int[5]'
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem2' 'const int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray2' 'const int[5]'

  void MemFunc1(int Arg) {
    // CHECK-NEXT: CXXMethodDecl{{.*}}MemFunc1 'void (int)'
    // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'int'
    // CHECK-NEXT: CompoundStmt
    int Local;
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} Local 'int'
    int LocalArray[5];
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} LocalArray 'int[5]'
#pragma acc declare present(Arg, Local, LocalArray)
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: OpenACCDeclareDecl
    // CHECK-NEXT: present clause
    // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'int'
    // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
    // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'int[5]'
  }
  void MemFunc2(int Arg);
  // CHECK: CXXMethodDecl{{.*}}MemFunc2
};
void Struct::MemFunc2(int Arg) {
  // CHECK: CXXMethodDecl{{.*}}MemFunc2 'void (int)'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'int'
  // CHECK-NEXT: CompoundStmt
  int Local;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} Local 'int'
  int LocalArray[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} LocalArray 'int[5]'
#pragma acc declare present(Arg, Local, LocalArray)
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'int[5]'
}

void NormalFunc(int Arg) {
  // CHECK-NEXT: FunctionDecl{{.*}}NormalFunc 'void (int)'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'int'
  // CHECK-NEXT: CompoundStmt
  int Local;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} Local 'int'
  int LocalArray[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} LocalArray 'int[5]'
#pragma acc declare present(Arg, Local, LocalArray)
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'int[5]'
}

template<typename T>
struct DependentStruct {
  // CHECK: ClassTemplateDecl{{.*}}DependentStruct
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 0 index 0 T
  // CHECK-NEXT: CXXRecordDecl{{.*}}DependentStruct definition
  // CHECK: CXXRecordDecl{{.*}}implicit struct DependentStruct
  static const T StaticMem = 5;
  // CHECK-NEXT: VarDecl{{.*}} StaticMem 'const T' static cinit
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 5
  static const T StaticMem2 = 5;
  // CHECK-NEXT: VarDecl{{.*}} StaticMem2 'const T' static cinit
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 5
  static constexpr T StaticMemArray[5] = {};
  // CHECK-NEXT: VarDecl{{.*}} StaticMemArray 'const T[5]'
  // CHECK-NEXT: InitListExpr{{.*}}'void'
  static constexpr T StaticMemArray2[5] = {};
  // CHECK-NEXT: VarDecl{{.*}} StaticMemArray2 'const T[5]'
  // CHECK-NEXT: InitListExpr{{.*}}'void'
#pragma acc declare copyin(StaticMem, StaticMemArray) create(zero: StaticMem2, StaticMemArray2)
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem' 'const T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray' 'const T[5]'
  // CHECK-NEXT: create clause modifiers: zero
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem2' 'const T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray2' 'const T[5]'

  template<typename U>
  void DepMemFunc1(U Arg, U Arg2) {
    // CHECK-NEXT: FunctionTemplateDecl{{.*}}DepMemFunc1
    // CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 1 index 0 U
    // CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc1 'void (U, U)'
    // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'U'
    // CHECK-NEXT: ParmVarDecl{{.*}} Arg2 'U'
    // CHECK-NEXT: CompoundStmt
    T Local, Local2;
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} Local 'T'
    // CHECK-NEXT: VarDecl{{.*}} Local2 'T'
    U LocalArray[5];
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} LocalArray 'U[5]'
    U LocalArray2[5];
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} LocalArray2 'U[5]'
#pragma acc declare copy(always, alwaysin: Arg, Local, LocalArray) copyout(zero: Arg2, Local2, LocalArray2)
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: OpenACCDeclareDecl
    // CHECK-NEXT: copy clause modifiers: always, alwaysin
    // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'U'
    // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'T'
    // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'U[5]'
    // CHECK-NEXT: copyout clause modifiers: zero
    // CHECK-NEXT: DeclRefExpr{{.*}}'Arg2' 'U'
    // CHECK-NEXT: DeclRefExpr{{.*}}'Local2' 'T'
    // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray2' 'U[5]'

    extern T Local3;
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} Local3 'T' extern
    T Local4;
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl{{.*}} Local4 'T'
#pragma acc declare link(Local3) device_resident(Local4)
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: OpenACCDeclareDecl
    // CHECK-NEXT: link clause
    // CHECK-NEXT: DeclRefExpr{{.*}}'Local3' 'T'
    // CHECK-NEXT: device_resident clause
    // CHECK-NEXT: DeclRefExpr{{.*}}'Local4' 'T'
  }
  template<typename U>
  void DepMemFunc2(U Arg);
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}DepMemFunc2
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 1 index 0 U
  // CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc2 'void (U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'U'
};

// Instantiation of class.
// CHECK-NEXT: ClassTemplateSpecializationDecl{{.*}}DependentStruct definition
// CHECK: TemplateArgument type 'int'
// CHECK-NEXT: BuiltinType{{.*}}'int'
// CHECK-NEXT: CXXRecordDecl{{.*}} struct DependentStruct

// CHECK-NEXT: VarDecl{{.*}} StaticMem 'const int' 
// CHECK-NEXT: IntegerLiteral{{.*}}'int' 5

// CHECK-NEXT: VarDecl{{.*}} StaticMem2 'const int' 
// CHECK-NEXT: IntegerLiteral{{.*}}'int' 5
//
// CHECK-NEXT: VarDecl{{.*}} StaticMemArray 'const int[5]'
// CHECK-NEXT: value: Array size=5
// CHECK-NEXT: filler: 5 x Int 0
// CHECK-NEXT: InitListExpr{{.*}} 'const int[5]'
// CHECK-NEXT: array_filler

// CHECK-NEXT: VarDecl{{.*}} StaticMemArray2 'const int[5]'
// CHECK-NEXT: value: Array size=5
// CHECK-NEXT: filler: 5 x Int 0
// CHECK-NEXT: InitListExpr{{.*}} 'const int[5]'
// CHECK-NEXT: array_filler

// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: copyin clause
// CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem' 'const int'
// CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray' 'const int[5]'
// CHECK-NEXT: create clause modifiers: zero
// CHECK-NEXT: DeclRefExpr{{.*}}'StaticMem2' 'const int'
// CHECK-NEXT: DeclRefExpr{{.*}}'StaticMemArray2' 'const int[5]'

// CHECK-NEXT: FunctionTemplateDecl{{.*}} DepMemFunc1
// CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 0 index 0 U
// CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc1 'void (U, U)'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg 'U'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg2 'U'
// CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc1 'void (float, float)'
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT: BuiltinType{{.*}}'float'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg 'float'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg2 'float'
// CHECK-NEXT: CompoundStmt

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} Local 'int'
// CHECK-NEXT: VarDecl{{.*}} Local2 'int'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} LocalArray 'float[5]'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} LocalArray2 'float[5]'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: copy clause modifiers: always, alwaysin
// CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'float'
// CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
// CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'float[5]'
// CHECK-NEXT: copyout clause modifiers: zero
// CHECK-NEXT: DeclRefExpr{{.*}}'Arg2' 'float'
// CHECK-NEXT: DeclRefExpr{{.*}}'Local2' 'int'
// CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray2' 'float[5]'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} Local3 'int' extern
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} Local4 'int'
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: link clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Local3' 'int'
// CHECK-NEXT: device_resident clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Local4' 'int'

// CHECK-NEXT: FunctionTemplateDecl{{.*}}DepMemFunc2
// CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 0 index 0 U
// CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc2 'void (U)'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg 'U'
// CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc2 'void (float)'
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT: BuiltinType{{.*}}'float'
// CHECK-NEXT: ParmVarDecl{{.*}} Arg 'float'
// CHECK-NEXT: CompoundStmt

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} Local 'int'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl{{.*}} LocalArray 'float[5]'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCDeclareDecl
// CHECK-NEXT: present clause
// CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'float'
// CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
// CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'float[5]'

template<typename T>
template<typename U>
void DependentStruct<T>::DepMemFunc2(U Arg) {
  // CHECK: FunctionTemplateDecl{{.*}} DepMemFunc2
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 1 index 0 U
  // CHECK-NEXT: CXXMethodDecl{{.*}}DepMemFunc2 'void (U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'U'
  // CHECK-NEXT: CompoundStmt
  T Local;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} Local 'T'
  U LocalArray[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} LocalArray 'U[5]'
#pragma acc declare present(Arg, Local, LocalArray)
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'U[5]'
}

template<typename T, unsigned Size>
void DependentFunc(T Arg) {
  // CHECK: FunctionTemplateDecl{{.*}} DependentFunc
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}depth 0 index 0 T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} depth 0 index 1 Size
  // CHECK-NEXT: FunctionDecl{{.*}}DependentFunc 'void (T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'T'
  // CHECK-NEXT: CompoundStmt
  T Local;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} Local 'T'
  T LocalArray[Size];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} LocalArray 'T[Size]'

#pragma acc declare present(Arg, Local, LocalArray)
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'T[Size]'

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} DependentFunc 'void (int)'
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}}'int'
  // CHECK-NEXT: TemplateArgument integral '5U'
  // CHECK-NEXT: ParmVarDecl{{.*}} Arg 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} Local 'int'

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} LocalArray 'int[5]'

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: OpenACCDeclareDecl
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Arg' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'Local' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'LocalArray' 'int[5]'
}

void use() {
  float i;
  DependentStruct<int> S;
  S.DepMemFunc1(i, i);
  S.DepMemFunc2(i);
  DependentFunc<int, 5>(i);
}

#endif // PCH_HELPER
