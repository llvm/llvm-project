// RUN: %clang_cc1 -triple i686-windows-msvc   -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck %s



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Declarations are not exported.
// CHECK-NOT: @ExternGlobalDecl
__declspec(dllexport) extern int ExternGlobalDecl;

// dllexport implies a definition.
// CHECK-DAG: @GlobalDef = common dllexport global i32 0, align 4
__declspec(dllexport) int GlobalDef;

// Export definition.
// CHECK-DAG: @GlobalInit = dllexport global i32 1, align 4
__declspec(dllexport) int GlobalInit = 1;

// Declare, then export definition.
// CHECK-DAG: @GlobalDeclInit = dllexport global i32 1, align 4
__declspec(dllexport) extern int GlobalDeclInit;
int GlobalDeclInit = 1;

// Redeclarations
// CHECK-DAG: @GlobalRedecl1 = common dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl1;
__declspec(dllexport)        int GlobalRedecl1;

// CHECK-DAG: @GlobalRedecl2 = common dllexport global i32 0, align 4
__declspec(dllexport) extern int GlobalRedecl2;
                             int GlobalRedecl2;



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Declarations are not exported.

// Export function definition.
// CHECK-DAG: define dllexport void @def()
__declspec(dllexport) void def(void) {}

// Export inline function.
// CHECK-DAG: define weak_odr dllexport void @inlineFunc()
// CHECK-DAG: define weak_odr dllexport void @externInlineFunc()
__declspec(dllexport) inline void inlineFunc(void) {}
__declspec(dllexport) inline void externInlineFunc(void) {}
extern void externInlineFunc(void);

// Redeclarations
// CHECK-DAG: define dllexport void @redecl1()
__declspec(dllexport) void redecl1(void);
__declspec(dllexport) void redecl1(void) {}

// CHECK-DAG: define dllexport void @redecl2()
__declspec(dllexport) void redecl2(void);
                      void redecl2(void) {}



//===----------------------------------------------------------------------===//
// Precedence
//===----------------------------------------------------------------------===//

// dllexport takes precedence over the dllimport if both are specified.
// CHECK-DAG: @PrecedenceGlobal1A = common dllexport global i32 0, align 4
// CHECK-DAG: @PrecedenceGlobal1B = common dllexport global i32 0, align 4
__attribute__((dllimport, dllexport))       int PrecedenceGlobal1A;
__declspec(dllimport) __declspec(dllexport) int PrecedenceGlobal1B;

// CHECK-DAG: @PrecedenceGlobal2A = common dllexport global i32 0, align 4
// CHECK-DAG: @PrecedenceGlobal2B = common dllexport global i32 0, align 4
__attribute__((dllexport, dllimport))       int PrecedenceGlobal2A;
__declspec(dllexport) __declspec(dllimport) int PrecedenceGlobal2B;

// CHECK-DAG: @PrecedenceGlobalRedecl1 = dllexport global i32 0, align 4
__declspec(dllexport) extern int PrecedenceGlobalRedecl1;
__declspec(dllimport)        int PrecedenceGlobalRedecl1 = 0;

// CHECK-DAG: @PrecedenceGlobalRedecl2 = common dllexport global i32 0, align 4
__declspec(dllimport) extern int PrecedenceGlobalRedecl2;
__declspec(dllexport)        int PrecedenceGlobalRedecl2;

// CHECK-DAG: @PrecedenceGlobalMixed1 = dllexport global i32 1, align 4
__attribute__((dllexport)) extern int PrecedenceGlobalMixed1;
__declspec(dllimport)             int PrecedenceGlobalMixed1 = 1;

// CHECK-DAG: @PrecedenceGlobalMixed2 = common dllexport global i32 0, align 4
__attribute__((dllimport)) extern int PrecedenceGlobalMixed2;
__declspec(dllexport)             int PrecedenceGlobalMixed2;

// CHECK-DAG: define dllexport void @precedence1A()
// CHECK-DAG: define dllexport void @precedence1B()
void __attribute__((dllimport, dllexport))       precedence1A(void) {}
void __declspec(dllimport) __declspec(dllexport) precedence1B(void) {}

// CHECK-DAG: define dllexport void @precedence2A()
// CHECK-DAG: define dllexport void @precedence2B()
void __attribute__((dllexport, dllimport))       precedence2A(void) {}
void __declspec(dllexport) __declspec(dllimport) precedence2B(void) {}

// CHECK-DAG: define dllexport void @precedenceRedecl1()
void __declspec(dllimport) precedenceRedecl1(void);
void __declspec(dllexport) precedenceRedecl1(void) {}

// CHECK-DAG: define dllexport void @precedenceRedecl2()
void __declspec(dllexport) precedenceRedecl2(void);
void __declspec(dllimport) precedenceRedecl2(void) {}
