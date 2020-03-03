// RUN: %clang_cc1 -triple x86_64-windows-msvc -emit-llvm -o - %s | FileCheck %s
typedef float TooLargeAlignment __attribute__((__vector_size__(64)));
typedef float NormalAlignment __attribute__((__vector_size__(4)));

TooLargeAlignment TooBig;
// CHECK: @TooBig = dso_local global <16 x float>  zeroinitializer, align 64
NormalAlignment JustRight;
// CHECK: @JustRight = common dso_local global <1 x float>  zeroinitializer, align 4

TooLargeAlignment *IsAPointer;
// CHECK: @IsAPointer = common dso_local global <16 x float>* null, align 8
