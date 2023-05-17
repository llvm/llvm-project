// RUN: %clang_cc1 -emit-llvm -o - -triple=i386-pc-win32 -std=c++11 %s -fcxx-exceptions -fms-extensions | FileCheck %s

// CHECK-DAG: @"??_R0?AUY@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUY@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUY@@@8??0Y@@QAE@ABU0@@Z8" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, ptr @"??_R0?AUY@@@8", i32 0, i32 -1, i32 0, i32 8, ptr @"??0Y@@QAE@ABU0@@Z" }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUZ@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUZ@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUZ@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUZ@@@8", i32 0, i32 -1, i32 0, i32 1, ptr null }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUW@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUW@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUW@@@8??0W@@QAE@ABU0@@Z44" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 4, ptr @"??_R0?AUW@@@8", i32 4, i32 -1, i32 0, i32 4, ptr @"??0W@@QAE@ABU0@@Z" }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUM@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUM@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUM@@@818" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUM@@@8", i32 8, i32 -1, i32 0, i32 1, ptr null }, section ".xdata", comdat
// CHECK-DAG: @"??_R0?AUV@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { ptr @"??_7type_info@@6B@", ptr null, [8 x i8] c".?AUV@@\00" }, comdat
// CHECK-DAG: @"_CT??_R0?AUV@@@81044" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUV@@@8", i32 0, i32 4, i32 4, i32 1, ptr null }, section ".xdata", comdat
// CHECK-DAG: @"_CTA5?AUY@@" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.5 { i32 5, [5 x ptr] [ptr @"_CT??_R0?AUY@@@8??0Y@@QAE@ABU0@@Z8", ptr @"_CT??_R0?AUZ@@@81", ptr @"_CT??_R0?AUW@@@8??0W@@QAE@ABU0@@Z44", ptr @"_CT??_R0?AUM@@@818", ptr @"_CT??_R0?AUV@@@81044"] }, section ".xdata", comdat
// CHECK-DAG: @"_TI5?AUY@@" = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, ptr @"??_DY@@QAEXXZ", ptr null, ptr @"_CTA5?AUY@@" }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUDefault@@@8??_ODefault@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUDefault@@@8", i32 0, i32 -1, i32 0, i32 1, ptr @"??_ODefault@@QAEXAAU0@@Z" }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUDeletedCopy@@@81" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUDeletedCopy@@@8", i32 0, i32 -1, i32 0, i32 1, ptr null }, section ".xdata", comdat
// CHECk-DAG: @"_CT??_R0?AUMoveOnly@@@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUMoveOnly@@@8", i32 0, i321-1, i32 0, i32 4, ptr null }, section ".xdata", comda
// CHECK-DAG: @"_CT??_R0?AUVariadic@@@8??_OVariadic@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUVariadic@@@8", i32 0, i32 -1, i32 0, i32 1, ptr @"??_OVariadic@@QAEXAAU0@@Z" }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0?AUTemplateWithDefault@@@8??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z1" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 0, ptr @"??_R0?AUTemplateWithDefault@@@8", i32 0, i32 -1, i32 0, i32 1, ptr @"??$?_OH@TemplateWithDefault@@QAEXAAU0@@Z" }, section ".xdata", comdat
// CHECK-DAG: @"_CTA2$$T" = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x ptr] [ptr @"_CT??_R0$$T@84", ptr @"_CT??_R0PAX@84"] }, section ".xdata", comdat
// CHECK-DAG: @"_CT??_R0P6AXXZ@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, ptr @"??_R0P6AXXZ@8", i32 0, i32 -1, i32 0, i32 4, ptr null }, section ".xdata", comdat
// CHECK-DAG: @_CTA1P6AXXZ = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x ptr] [ptr @"_CT??_R0P6AXXZ@84"] }, section ".xdata", comdat
// CHECK-DAG: @_TI1P6AXXZ = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, ptr null, ptr null, ptr @_CTA1P6AXXZ }, section ".xdata", comdat
// CHECK-DAG: @_TIU2PAPFAH = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 4, ptr null, ptr null, ptr @_CTA2PAPFAH }, section ".xdata", comdat
// CHECK-DAG: @_CTA2PAPFAH = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.2 { i32 2, [2 x ptr] [ptr @"_CT??_R0PAPFAH@84", ptr @"_CT??_R0PAX@84"] }, section ".xdata", comdat
// CHECK-DAG: @"_TI1?AUFoo@?A0x{{[^@]*}}@@" = internal unnamed_addr constant %eh.ThrowInfo { i32 0, ptr null, ptr null, ptr @"_CTA1?AUFoo@?A0x{{[^@]*}}@@" }, section ".xdata"


struct N { ~N(); };
struct M : private N {};
struct X {};
struct Z {};
struct V : private X {};
struct W : M, virtual V {};
struct Y : Z, W, virtual V {};

void f(const Y &y) {
  // CHECK-LABEL: @"?f@@YAXABUY@@@Z"
  // CHECK: call x86_thiscallcc noundef ptr @"??0Y@@QAE@ABU0@@Z"(ptr {{[^,]*}} %[[mem:.*]], ptr
  // CHECK: call void @_CxxThrowException(ptr %[[mem]], ptr @"_TI5?AUY@@")
  throw y;
}

void g(const int *const *y) {
  // CHECK-LABEL: @"?g@@YAXPBQBH@Z"
  // CHECK: call void @_CxxThrowException(ptr %{{.*}}, ptr @_TIC2PAPBH)
  throw y;
}

void h(__unaligned int * __unaligned *y) {
  // CHECK-LABEL: @"?h@@YAXPFAPFAH@Z"
  // CHECK: call void @_CxxThrowException(ptr %{{.*}}, ptr @_TIU2PAPFAH)
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
// CHECK: call x86_thiscallcc {{.*}} @"??0Default@@QAE@AAU0@H@Z"({{.*}} %[[this]], {{.*}} %[[src]], i32 noundef 42)
// CHECK: ret void

void h(Default &d) {
  throw d;
}

struct DeletedCopy {
  DeletedCopy();
  DeletedCopy(DeletedCopy &&);
  DeletedCopy(const DeletedCopy &) = delete;
};
void throwDeletedCopy() { throw DeletedCopy(); }


struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &&);
  ~MoveOnly();
  MoveOnly(const MoveOnly &) = delete;

  // For some reason this subobject was important for reproducing PR43680
  struct HasCopy {
    HasCopy();
    HasCopy(const HasCopy &o);
    ~HasCopy();
    int x;
  } sub;
};

void throwMoveOnly() { throw MoveOnly(); }

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

namespace std {
template <typename T>
void *__GetExceptionInfo(T);
}
using namespace std;

void *GetExceptionInfo_test0() {
// CHECK-LABEL: @"?GetExceptionInfo_test0@@YAPAXXZ"
// CHECK:  ret ptr @_TI1H
  return __GetExceptionInfo(0);
}

void *GetExceptionInfo_test1() {
// CHECK-LABEL: @"?GetExceptionInfo_test1@@YAPAXXZ"
// CHECK:  ret ptr @_TI1P6AXXZ
  return __GetExceptionInfo<void (*)()>(&h);
}

// PR36327: Try an exception type with no linkage.
namespace { struct Foo { } foo_exc; }

void *GetExceptionInfo_test2() {
// CHECK-LABEL: @"?GetExceptionInfo_test2@@YAPAXXZ"
// CHECK:  ret ptr @"_TI1?AUFoo@?A0x{{[^@]*}}@@"
  return __GetExceptionInfo(foo_exc);
}
