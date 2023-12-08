// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-elf -DCF_BUILDING_CF -DDECL -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DECL
// RUN: %clang_cc1 -triple x86_64-elf -DCF_BUILDING_CF -DDEFN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DEFN
// RUN: %clang_cc1 -triple x86_64-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF
// RUN: %clang_cc1 -triple x86_64-elf -DEXTERN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN

// RUN: %clang_cc1 -Os -triple x86_64-elf -DCF_BUILDING_CF -DDECL -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DECL
// RUN: %clang_cc1 -Os -triple x86_64-elf -DCF_BUILDING_CF -DDEFN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-IN-CF-DEFN
// RUN: %clang_cc1 -Os -triple x86_64-elf -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF
// RUN: %clang_cc1 -Os -triple x86_64-elf -DEXTERN -S -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-CF-EXTERN


#if defined(CF_BUILDING_CF)
#if defined(DECL)
extern long __CFConstantStringClassReference[];
#elif defined(DEFN)
long __CFConstantStringClassReference[32];
#endif
#else
#if defined(EXTERN)
extern long __CFConstantStringClassReference[];
#else
long __CFConstantStringClassReference[];
#endif
#endif

typedef struct __CFString *CFStringRef;
const CFStringRef string = (CFStringRef)__builtin___CFStringMakeConstantString("string");


// CHECK-CF-IN-CF-DECL: @__CFConstantStringClassReference = external global [0 x i32]
// CHECK-CF-IN-CF-DEFN: @__CFConstantStringClassReference ={{.*}} global [32 x i64] zeroinitializer, align 16
// CHECK-CF: @__CFConstantStringClassReference ={{.*}} global [1 x i64] zeroinitializer, align 8
// CHECK-CF-EXTERN: @__CFConstantStringClassReference = external global [0 x i32]
// CHECK-CF-EXTERN: @.str = private unnamed_addr constant [7 x i8] c"string\00", section ".rodata", align 1
