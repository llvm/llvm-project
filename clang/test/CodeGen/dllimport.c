// RUN: %clang_cc1 -triple i686-windows-msvc   -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck --check-prefix=CHECK --check-prefix=GNU %s
// RUN: %clang_cc1 -triple x86_64-windows-gnu  -fms-extensions -emit-llvm -std=c11 -O0 -o - %s | FileCheck --check-prefix=CHECK --check-prefix=GNU %s
// RUN: %clang_cc1 -triple i686-windows-msvc   -fms-extensions -emit-llvm -std=c11 -O1 -fno-experimental-new-pass-manager -o - %s | FileCheck --check-prefix=O1 --check-prefix=MO1 %s
// RUN: %clang_cc1 -triple i686-windows-gnu    -fms-extensions -emit-llvm -std=c11 -O1 -fno-experimental-new-pass-manager -o - %s | FileCheck --check-prefix=O1 --check-prefix=GO1 %s

#define JOIN2(x, y) x##y
#define JOIN(x, y) JOIN2(x, y)
#define USEVAR(var) int JOIN(use, __LINE__)() { return var; }
#define USE(func) void JOIN(use, __LINE__)() { func(); }



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
// CHECK: @ExternGlobalDecl = external dllimport global i32
__declspec(dllimport) extern int ExternGlobalDecl;
USEVAR(ExternGlobalDecl)

// dllimport implies a declaration.
// CHECK: @GlobalDecl = external dllimport global i32
__declspec(dllimport) int GlobalDecl;
USEVAR(GlobalDecl)

// Redeclarations
// CHECK: @GlobalRedecl1 = external dllimport global i32
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;
USEVAR(GlobalRedecl1)

// CHECK: @GlobalRedecl2 = external dllimport global i32
__declspec(dllimport) int GlobalRedecl2;
__declspec(dllimport) int GlobalRedecl2;
USEVAR(GlobalRedecl2)

// NB: MSVC issues a warning and makes GlobalRedecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// MS: @GlobalRedecl3 = external dso_local global i32
// GNU: @GlobalRedecl3 = external global i32
__declspec(dllimport) extern int GlobalRedecl3;
                      extern int GlobalRedecl3; // dllimport ignored
USEVAR(GlobalRedecl3)

// Make sure this works even if the decl has been used before it's defined (PR20792).
// MS: @GlobalRedecl4 = common dso_local dllexport global i32
// GNU: @GlobalRedecl4 = common dso_local global i32
__declspec(dllimport) extern int GlobalRedecl4;
USEVAR(GlobalRedecl4)
                      int GlobalRedecl4; // dllimport ignored

// FIXME: dllimport is dropped in the AST; this should be reflected in codegen (PR02803).
// CHECK: @GlobalRedecl5 = external dllimport global i32
__declspec(dllimport) extern int GlobalRedecl5;
USEVAR(GlobalRedecl5)
                      extern int GlobalRedecl5; // dllimport ignored

// Redeclaration in local context.
// CHECK: @GlobalRedecl6 = external dllimport global i32
__declspec(dllimport) int GlobalRedecl6;
int functionScope() {
  extern int GlobalRedecl6; // still dllimport
  return GlobalRedecl6;
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration.
// CHECK-DAG: declare dllimport void @decl()
__declspec(dllimport) void decl(void);

// Initialize use_decl with the address of the thunk.
// CHECK-DAG: @use_decl = dso_local global void ()* @decl
void (*use_decl)(void) = &decl;

// Import inline function.
// MS-DAG: declare dllimport void @inlineFunc()
// MO1-DAG: define available_externally dllimport void @inlineFunc()
// GNU-DAG: declare dso_local void @inlineFunc()
// GO1-DAG: define available_externally dso_local void @inlineFunc()
__declspec(dllimport) inline void inlineFunc(void) {}
USE(inlineFunc)

// inline attributes
// MS-DAG: declare dllimport void @noinline()
// MO1-DAG: define available_externally dllimport void @noinline()
// GNU-DAG: declare dso_local void @noinline()
// GO1-DAG: define available_externally dso_local void @noinline()
// CHECK-NOT: @alwaysInline()
// O1-NOT: @alwaysInline()
__declspec(dllimport) __attribute__((noinline)) inline void noinline(void) {}
__declspec(dllimport) __attribute__((always_inline)) inline void alwaysInline(void) {}
USE(noinline)
USE(alwaysInline)

// Redeclarations
// CHECK-DAG: declare dllimport void @redecl1()
__declspec(dllimport) void redecl1(void);
__declspec(dllimport) void redecl1(void);
USE(redecl1)

// NB: MSVC issues a warning and makes redecl2/redecl3 dllexport. We follow GCC
// and drop the dllimport with a warning.
// CHECK-DAG: declare dso_local void @redecl2()
__declspec(dllimport) void redecl2(void);
                      void redecl2(void);
USE(redecl2)

// MS: define dso_local dllexport void @redecl3()
// GNU: define dso_local void @redecl3()
__declspec(dllimport) void redecl3(void);
                      void redecl3(void) {} // dllimport ignored
USE(redecl3)

// Make sure this works even if the decl is used before it's defined (PR20792).
// MS: define dso_local dllexport void @redecl4()
// GNU: define dso_local void @redecl4()
__declspec(dllimport) void redecl4(void);
USE(redecl4)
                      void redecl4(void) {} // dllimport ignored

// FIXME: dllimport is dropped in the AST; this should be reflected in codegen (PR20803).
// CHECK-DAG: declare dllimport
__declspec(dllimport) void redecl5(void);
USE(redecl5)
                      void redecl5(void); // dllimport ignored
