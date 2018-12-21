// RUN: %clang_cc1 %s -emit-llvm -triple i686-windows-gnu -o - | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -triple i686-windows-itanium -o - | FileCheck %s

// GCC 5.1 began mangling these Windows calling conventions into function
// types, since they can be used for overloading. They've always been mangled
// in the MS ABI, but they are new to the Itanium mangler. Note that the main
// function definition does not use a calling convention. Only function types
// that appear later use it.

template <typename Fn> static int func_as_ptr(Fn fn) { return int(fn); }

void f_cdecl(int, int);
void __attribute__((stdcall)) f_stdcall(int, int);
void __attribute__((fastcall)) f_fastcall(int, int);
void __attribute__((thiscall)) f_thiscall(int, int);

int as_cdecl() { return func_as_ptr(f_cdecl); }
int as_stdcall() { return func_as_ptr(f_stdcall); }
int as_fastcall() { return func_as_ptr(f_fastcall); }

// CHECK: define dso_local i32 @_Z8as_cdeclv()
// CHECK:   call i32 @_ZL11func_as_ptrIPFviiEEiT_(void (i32, i32)* @_Z7f_cdeclii)

// CHECK: define dso_local i32 @_Z10as_stdcallv()
// CHECK:   call i32 @_ZL11func_as_ptrIPU7stdcallFviiEEiT_(void (i32, i32)* @"\01__Z9f_stdcallii@8")

// CHECK: define dso_local i32 @_Z11as_fastcallv()
// CHECK:   call i32 @_ZL11func_as_ptrIPU8fastcallFviiEEiT_(void (i32, i32)* @"\01@_Z10f_fastcallii@8")

// PR40107: We should mangle thiscall here but we don't because we can't
// disambiguate it from the member pointer case below where it shouldn't be
// mangled.
//int as_thiscall() { return func_as_ptr(f_thiscall); }
// CHECKX: define dso_local i32 @_Z11as_thiscallv()
// CHECKX:   call i32 @_ZL11func_as_ptrIPU8thiscallFviiEEiT_(void (i32, i32)* @_Z10f_thiscallii)

// CHECK: define dso_local void @_Z11funcRefTypeRU8fastcallFviiE(void (i32, i32)* %fr)
void funcRefType(void(__attribute__((fastcall)) & fr)(int, int)) {
  fr(1, 2);
}

struct Foo { void bar(int, int); };

// PR40107: In this case, the member function pointer uses the thiscall
// convention, but GCC doesn't mangle it, so we don't either.
// CHECK: define dso_local void @_Z15memptr_thiscallP3FooMS_FvvE(%struct.Foo* {{.*}})
void memptr_thiscall(Foo *o, void (Foo::*mp)()) { (o->*mp)(); }

// CHECK: define dso_local void @_Z12memptrCCTypeR3FooMS_U8fastcallFviiE(%struct.Foo* {{.*}}, { i32, i32 }* byval{{.*}})
void memptrCCType(Foo &o, void (__attribute__((fastcall)) Foo::*mp)(int, int)) {
  (o.*mp)(1, 2);
}

// CHECK: define dso_local i32 @_Z17useTemplateFnTypev()
// CHECK:   call i32 @_ZL14templateFnTypeIU8fastcallFviiEElPT_(void (i32, i32)* @"\01@_Z10f_fastcallii@8")
template <typename Fn> static long templateFnType(Fn *fn) { return long(fn); }
long useTemplateFnType() { return templateFnType(f_fastcall); }

// CHECK: define weak_odr dso_local x86_fastcallcc void @"\01@_Z10fnTemplateIsEvv@0"()
// CHECK: define          dso_local x86_fastcallcc void @"\01@_Z10fnTemplateIiEvv@0"()
template <typename T> void __attribute__((fastcall)) fnTemplate() {}
template void __attribute__((fastcall)) fnTemplate<short>();
template <> void __attribute__((fastcall)) fnTemplate<int>() {}

// CHECK: define weak_odr dso_local x86_fastcallcc void (i32, i32)* @"\01@_Z12fnTempReturnIsEPU8fastcallFviiEv@0"()
// CHECK: define          dso_local x86_fastcallcc void (i32, i32)* @"\01@_Z12fnTempReturnIiEPU8fastcallFviiEv@0"()
typedef void (__attribute__((fastcall)) *fp_cc_t)(int, int);
template <typename T> fp_cc_t __attribute__((fastcall)) fnTempReturn() { return nullptr; }
template fp_cc_t __attribute__((fastcall)) fnTempReturn<short>();
template <> fp_cc_t __attribute__((fastcall)) fnTempReturn<int>() { return nullptr; }
