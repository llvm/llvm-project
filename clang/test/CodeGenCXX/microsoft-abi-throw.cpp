// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 -std=c++11 %s -fcxx-exceptions -fms-extensions | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 -std=c++11 %s -fcxx-exceptions -fms-extensions -DSTD | FileCheck %s

// CHECK-DAG: @"??_R0?AUY@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"??_7type_info@@6B@", i8* null, [8 x i8] c".?AUY@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUY@@@8??0Y@@QAE@ABU0@@Z8" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUY@@@8" to i8*), i32 0, i32 -1, i32 0, i32 8, i8* bitcast (%struct.Y* (%struct.Y*, %struct.Y*, i32)* @"??0Y@@QAE@ABU0@@Z" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUZ@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"??_7type_info@@6B@", i8* null, [8 x i8] c".?AUZ@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUZ@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUZ@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* null }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUW@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"??_7type_info@@6B@", i8* null, [8 x i8] c".?AUW@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUW@@@8??0W@@QAE@ABU0@@Z44" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUW@@@8" to i8*), i32 4, i32 -1, i32 0, i32 4, i8* bitcast (%struct.W* (%struct.W*, %struct.W*, i32)* @"??0W@@QAE@ABU0@@Z" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUM@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"??_7type_info@@6B@", i8* null, [8 x i8] c".?AUM@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUM@@@818" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUM@@@8" to i8*), i32 8, i32 -1, i32 0, i32 1, i8* null }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUV@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"??_7type_info@@6B@", i8* null, [8 x i8] c".?AUV@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUV@@@81044" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0?AUV@@@8" to i8*), i32 0, i32 4, i32 4, i32 1, i8* null }, section ".xdata", comdat
// CHECK-DAG: @"_CTA5?AUY@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.5 { i32 5, [5 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0?AUY@@@8??0Y@@QAE@ABU0@@Z8", %eh.CatchableType* @"_CT??_R0?AUZ@@@81", %eh.CatchableType* @"_CT??_R0?AUW@@@8??0W@@QAE@ABU0@@Z44", %eh.CatchableType* @"_CT??_R0?AUM@@@818", %eh.CatchableType* @"_CT??_R0?AUV@@@81044"] }, section ".xdata", comdat
// CHECK-DAG: @"_TI5?AUY@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i8* bitcast (void (%struct.Y*)* @"??_DY@@QAEXXZ" to i8*), i8* null, i8* bitcast (%eh.CatchableTypeArray.5* @"_CTA5?AUY@@" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUDefault@@@8??_ODefault@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor13* @"??_R0?AUDefault@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* bitcast (void (%struct.Default*, %struct.Default*)* @"??_ODefault@@QAEXAAU0@@Z" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUVariadic@@@8??_OVariadic@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor14* @"??_R0?AUVariadic@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* bitcast (void (%struct.Variadic*, %struct.Variadic*)* @"??_OVariadic@@QAEXAAU0@@Z" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUTemplateWithDefault@@@8??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, i8* bitcast (%rtti.TypeDescriptor25* @"??_R0?AUTemplateWithDefault@@@8" to i8*), i32 0, i32 -1, i32 0, i32 1, i8* bitcast (void (%struct.TemplateWithDefault*, %struct.TemplateWithDefault*)* @"??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z" to i8*) }, section ".xdata", comdat
// CHECK-DAG: @"_CTA2$$T" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0$$T@84", %eh.CatchableType* @"_CT??_R0PAX@84"] }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0P6AXXZ@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i8* bitcast (%rtti.TypeDescriptor7* @"??_R0P6AXXZ@8" to i8*), i32 0, i32 -1, i32 0, i32 4, i8* null }, section ".xdata", comdat
// CHECK-DAG: @_CTA1P6AXXZ = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0P6AXXZ@84"] }, section ".xdata", comdat
// CHECK-DAG: @_TI1P6AXXZ = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i8* null, i8* null, i8* bitcast (%eh.CatchableTypeArray.1* @_CTA1P6AXXZ to i8*) }, section ".xdata", comdat
// CHECK-DAG: @_TIU2PAPFAH = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 4, i8* null, i8* null, i8* bitcast (%eh.CatchableTypeArray.2* @_CTA2PAPFAH to i8*) }, section ".xdata", comdat
// CHECK-DAG: @_CTA2PAPFAH = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0PAPFAH@84", %eh.CatchableType* @"_CT??_R0PAX@84"] }, section ".xdata", comdat
// CHECK-DAG: @"_TI1?AUFoo@?A0x{{[^@]*}}@@" = internal unnamed_addr constant %eh.ThrowInfo { i32 0, i8* null, i8* null, i8* bitcast (%eh.CatchableTypeArray.1* @"_CTA1?AUFoo@?A0x{{[^@]*}}@@" to i8*) }, section ".xdata"


struct N { ~N(); };
struct M : private N {};
struct X {};
struct Z {};
struct V : private X {};
struct W : M, virtual V {};
struct Y : Z, W, virtual V {};

void f(const Y &y) {
  // CHECK-LABEL: @"?f@@YAXABUY@@@Z"
  // CHECK: call x86_thiscallcc %struct.Y* @"??0Y@@QAE@ABU0@@Z"(%struct.Y* %[[mem:.*]], %struct.Y*
  // CHECK: %[[cast:.*]] = bitcast %struct.Y* %[[mem]] to i8*
  // CHECK: call void @_CxxThrowException(i8* %[[cast]], %eh.ThrowInfo* @"_TI5?AUY@@")
  throw y;
}

void g(const int *const *y) {
  // CHECK-LABEL: @"?g@@YAXPBQBH@Z"
  // CHECK: call void @_CxxThrowException(i8* %{{.*}}, %eh.ThrowInfo* @_TIC2PAPBH)
  throw y;
}

void h(__unaligned int * __unaligned *y) {
  // CHECK-LABEL: @"?h@@YAXPFAPFAH@Z"
  // CHECK: call void @_CxxThrowException(i8* %{{.*}}, %eh.ThrowInfo* @_TIU2PAPFAH)
  throw y;
}

struct Default {
  Default(Default &, int = 42);
};

// CHECK-LABEL: @"??_ODefault@@QAEXAAU0@@Z"
// CHECK: %[[src_addr:.*]] = alloca
// CHECK: %[[this_addr:.*]] = alloca
// CHECK: store {{.*}} %src, {{.*}} %[[src_addr]], align 4
// CHECK: store {{.*}} %this, {{.*}} %[[this_addr]], align 4
// CHECK: %[[this:.*]] = load {{.*}} %[[this_addr]]
// CHECK: %[[src:.*]] = load {{.*}} %[[src_addr]]
// CHECK: call x86_thiscallcc {{.*}} @"??0Default@@QAE@AAU0@H@Z"({{.*}} %[[this]], {{.*}} %[[src]], i32 42)
// CHECK: ret void

void h(Default &d) {
  throw d;
}

struct Variadic {
  Variadic(Variadic &, ...);
};

void i(Variadic &v) {
  throw v;
}

// CHECK-LABEL: @"??_OVariadic@@QAEXAAU0@@Z"
// CHECK:  %[[src_addr:.*]] = alloca
// CHECK:  %[[this_addr:.*]] = alloca
// CHECK:  store {{.*}} %src, {{.*}} %[[src_addr:.*]], align
// CHECK:  store {{.*}} %this, {{.*}} %[[this_addr:.*]], align
// CHECK:  %[[this:.*]] = load {{.*}} %[[this_addr]]
// CHECK:  %[[src:.*]] = load {{.*}} %[[src_addr]]
// CHECK:  call {{.*}} @"??0Variadic@@QAA@AAU0@ZZ"({{.*}} %[[this]], {{.*}} %[[src]])
// CHECK:  ret void

struct TemplateWithDefault {
  template <typename T>
  static int f() {
    return 0;
  }
  template <typename T = int>
  TemplateWithDefault(TemplateWithDefault &, T = f<T>());
};

void j(TemplateWithDefault &twd) {
  throw twd;
}


void h() {
  throw nullptr;
}

#ifdef STD
namespace std {
template <typename T>
void *__GetExceptionInfo(T);
}
#else
template <typename T>
void *__GetExceptionInfo(T);
#endif
using namespace std;

void *GetExceptionInfo_test0() {
// CHECK-LABEL: @"?GetExceptionInfo_test0@@YAPAXXZ"
// CHECK:  ret i8* bitcast (%eh.ThrowInfo* @_TI1H to i8*)
  return __GetExceptionInfo(0);
}

void *GetExceptionInfo_test1() {
// CHECK-LABEL: @"?GetExceptionInfo_test1@@YAPAXXZ"
// CHECK:  ret i8* bitcast (%eh.ThrowInfo* @_TI1P6AXXZ to i8*)
  return __GetExceptionInfo<void (*)()>(&h);
}

// PR36327: Try an exception type with no linkage.
namespace { struct Foo { } foo_exc; }

void *GetExceptionInfo_test2() {
// CHECK-LABEL: @"?GetExceptionInfo_test2@@YAPAXXZ"
// CHECK:  ret i8* bitcast (%eh.ThrowInfo* @"_TI1?AUFoo@?A0x{{[^@]*}}@@" to i8*)
  return __GetExceptionInfo(foo_exc);
}
