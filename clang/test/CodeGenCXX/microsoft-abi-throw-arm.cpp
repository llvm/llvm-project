// On 32-bit Arm Windows the throw information is image-relative, as on 64-bit
// targets, and the RTTI stays absolute. 32-bit x86 keeps absolute pointers.

// RUN: %clang_cc1 -emit-llvm -o - -triple=thumbv7-windows-msvc -std=c++17 %s -fcxx-exceptions -fexceptions -fms-extensions | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 -emit-llvm -o - -triple=i686-windows-msvc -std=c++17 %s -fcxx-exceptions -fexceptions -fms-extensions | FileCheck %s --check-prefix=X86 --implicit-check-not=__ImageBase

// ARM-DAG: %eh.CatchableType = type { i32, i32, i32, i32, i32, i32, i32 }
// ARM-DAG: %eh.CatchableTypeArray.1 = type { i32, [1 x i32] }
// ARM-DAG: %eh.CatchableTypeArray.2 = type { i32, [2 x i32] }
// ARM-DAG: %eh.ThrowInfo = type { i32, i32, i32, i32 }
// ARM-DAG: %rtti.CompleteObjectLocator = type { i32, i32, i32, ptr, ptr }
// ARM-DAG: %rtti.ClassHierarchyDescriptor = type { i32, i32, i32, ptr }
// ARM-DAG: %rtti.BaseClassDescriptor = type { ptr, i32, i32, i32, i32, i32, ptr }

// ARM-DAG: @"_CT??_R0?AUD@@@8??0D@@QAA@ABU0@@Z4" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i32 sub nuw nsw (i32 ptrtoint (ptr @"??_R0?AUD@@@8" to i32), i32 ptrtoint (ptr @__ImageBase to i32)), i32 0, i32 -1, i32 0, i32 4, i32 sub nuw nsw (i32 ptrtoint (ptr @"??0D@@QAA@ABU0@@Z" to i32), i32 ptrtoint (ptr @__ImageBase to i32)) }, section ".xdata", comdat
// ARM-DAG: @"_CTA2?AUD@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x i32] [i32 sub nuw nsw (i32 ptrtoint (ptr @"_CT??_R0?AUD@@@8??0D@@QAA@ABU0@@Z4" to i32), i32 ptrtoint (ptr @__ImageBase to i32)), i32 sub nuw nsw (i32 ptrtoint (ptr @"_CT??_R0?AUB@@@8??0B@@QAA@ABU0@@Z4" to i32), i32 ptrtoint (ptr @__ImageBase to i32))] }, section ".xdata", comdat
// ARM-DAG: @"_TI2?AUD@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 sub nuw nsw (i32 ptrtoint (ptr @"??1D@@UAA@XZ" to i32), i32 ptrtoint (ptr @__ImageBase to i32)), i32 0, i32 sub nuw nsw (i32 ptrtoint (ptr @"_CTA2?AUD@@" to i32), i32 ptrtoint (ptr @__ImageBase to i32)) }, section ".xdata", comdat
// ARM-DAG: @"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i32 sub nuw nsw (i32 ptrtoint (ptr @"??_R0H@8" to i32), i32 ptrtoint (ptr @__ImageBase to i32)), i32 0, i32 -1, i32 0, i32 4, i32 0 }, section ".xdata", comdat
// ARM-DAG: @_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i32 0, i32 0, i32 sub nuw nsw (i32 ptrtoint (ptr @_CTA1H to i32), i32 ptrtoint (ptr @__ImageBase to i32)) }, section ".xdata", comdat
// ARM-DAG: @"??_R4B@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 0, i32 0, i32 0, ptr @"??_R0?AUB@@@8", ptr @"??_R3B@@8" }, comdat
// ARM-DAG: @"??_R1A@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { ptr @"??_R0?AUB@@@8", i32 0, i32 0, i32 -1, i32 0, i32 64, ptr @"??_R3B@@8" }, comdat
// ARM-DAG: @__ImageBase = external dso_local constant i8

// X86-DAG: %eh.CatchableType = type { i32, ptr, i32, i32, i32, i32, ptr }
// X86-DAG: %eh.CatchableTypeArray.2 = type { i32, [2 x ptr] }
// X86-DAG: %eh.ThrowInfo = type { i32, ptr, ptr, ptr }
// X86-DAG: @"_TI2?AUD@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, ptr @"??1D@@UAE@XZ", ptr null, ptr @"_CTA2?AUD@@" }, section ".xdata", comdat
// X86-DAG: @"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, ptr @"??_R0H@8", i32 0, i32 -1, i32 0, i32 4, ptr null }, section ".xdata", comdat

struct B {
  virtual ~B();
};
struct D : B {
  D();
  D(const D &);
  ~D();
};
B::~B() {}

// The throw still passes the ThrowInfo by address.
// ARM-LABEL: define dso_local arm_aapcs_vfpcc void @"?throwD@@YAXABUD@@@Z"
// ARM: call arm_aapcs_vfpcc void @_CxxThrowException(ptr %{{.*}}, ptr @"_TI2?AUD@@")
// X86-LABEL: define dso_local void @"?throwD@@YAXABUD@@@Z"
// X86: call void @_CxxThrowException(ptr %{{.*}}, ptr @"_TI2?AUD@@")
void throwD(const D &d) { throw d; }

// ARM-LABEL: define dso_local arm_aapcs_vfpcc void @"?throwInt@@YAXXZ"
// ARM: call arm_aapcs_vfpcc void @_CxxThrowException(ptr %{{.*}}, ptr @_TI1H)
void throwInt() { throw 1; }
