// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -DCF_BUILDING_CF -DDECL -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DECL
// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -DCF_BUILDING_CF -DDEFN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DEFN
// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF
// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -DEXTERN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN
// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -DEXTERN_DLLIMPORT -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -triple thumbv7-windows -fdeclspec -DDLLIMPORT -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-DLLIMPORT

// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -DCF_BUILDING_CF -DDECL -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DECL
// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -DCF_BUILDING_CF -DDEFN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DEFN
// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF
// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -DEXTERN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN
// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -DEXTERN_DLLIMPORT -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN-DLLIMPORT
// RUN: %clang_cc1 -Os -triple thumbv7-windows -fdeclspec -DDLLIMPORT -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-DLLIMPORT

#if defined(CF_BUILDING_CF)
#if defined(DECL)
extern __declspec(dllexport) long __CFConstantStringClassReference[];
#elif defined(DEFN)
__declspec(dllexport) long __CFConstantStringClassReference[32];
#endif
#else
#if defined(EXTERN)
extern long __CFConstantStringClassReference[];
#elif defined(EXTERN_DLLIMPORT)
extern __declspec(dllimport) long __CFConstantStringClassReference[];
#elif defined(DLLIMPORT)
__declspec(dllimport) long __CFConstantStringClassReference[];
#endif
#endif

typedef struct __CFString *CFStringRef;
const CFStringRef string = (CFStringRef)__builtin___CFStringMakeConstantString("string");

// CHECK-CF-IN-CF-DECL: @__CFConstantStringClassReference = external dso_local dllexport global [0 x i32]
// CHECK-CF-IN-CF-DEFN: @__CFConstantStringClassReference = dso_local dllexport global [32 x i32]
// CHECK-CF: @__CFConstantStringClassReference = external dllimport global [0 x i32]
// CHECK-CF-EXTERN: @__CFConstantStringClassReference = external dllimport global [0 x i32]
// CHECK-CF-EXTERN-DLLIMPORT: @__CFConstantStringClassReference = external dllimport global [0 x i32]
// CHECK-CF-DLLIMPORT: @__CFConstantStringClassReference = external dllimport global [0 x i32]

