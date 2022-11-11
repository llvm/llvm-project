// RUN: %clang_cc1 -fms-extensions -fms-compatibility-version=19.20 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -fms-extensions -fms-compatibility-version=19.20 -triple aarch64-windows-msvc -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-windows-itanium -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple aarch64-windows-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s --check-prefix=GNU

void foo1() { throw 1; }
// _CxxThrowException should not be marked dllimport.
// MSVC-LABEL: define dso_local void @"?foo1@@YAXXZ"
// MSVC: call void @_CxxThrowException
// MSVC: declare dso_local void @_CxxThrowException(ptr, ptr)

// __cxa_throw should be marked dllimport for *-windows-itanium.
// ITANIUM-LABEL: define dso_local void @_Z4foo1v()
// ITANIUM: call void @__cxa_throw({{.*}})
// ITANIUM: declare dllimport void @__cxa_throw({{.*}})

// ... but not for *-windows-gnu.
// GNU-LABEL: define dso_local void @_Z4foo1v()
// GNU: call void @__cxa_throw({{.*}})
// GNU: declare dso_local void @__cxa_throw({{.*}})


void bar();
void foo2() noexcept(true) { bar(); }
// __std_terminate should not be marked dllimport.
// MSVC-LABEL: define dso_local void @"?foo2@@YAXXZ"
// MSVC: call void @__std_terminate()
// MSVC: declare dso_local void @__std_terminate()

// _ZSt9terminatev and __cxa_begin_catch should be marked dllimport.
// ITANIUM-LABEL: define linkonce_odr hidden void @__clang_call_terminate(ptr %0)
// ITANIUM: call ptr @__cxa_begin_catch({{.*}})
// ITANIUM: call void @_ZSt9terminatev()
// ITANIUM: declare dllimport ptr @__cxa_begin_catch(ptr)
// ITANIUM: declare dllimport void @_ZSt9terminatev()

// .. not for mingw.
// GNU-LABEL: define linkonce_odr hidden void @__clang_call_terminate(ptr %0)
// GNU: call ptr @__cxa_begin_catch({{.*}})
// GNU: call void @_ZSt9terminatev()
// GNU: declare dso_local ptr @__cxa_begin_catch(ptr)
// GNU: declare dso_local void @_ZSt9terminatev()


struct A {};
struct B { virtual void f(); };
struct C : A, virtual B {};
struct T {};
T *foo3() { return dynamic_cast<T *>((C *)0); }
// __RTDynamicCast should not be marked dllimport.
// MSVC-LABEL: define dso_local noundef ptr @"?foo3@@YAPEAUT@@XZ"
// MSVC: call ptr @__RTDynamicCast({{.*}})
// MSVC: declare dso_local ptr @__RTDynamicCast(ptr, i32, ptr, ptr, i32)

// Again, imported
// ITANIUM-LABEL: define dso_local noundef ptr @_Z4foo3v()
// ITANIUM: call ptr @__dynamic_cast({{.*}})
// ITANIUM: declare dllimport ptr @__dynamic_cast({{.*}})

// Not imported
// GNU-LABEL: define dso_local noundef ptr @_Z4foo3v()
// GNU: call ptr @__dynamic_cast({{.*}})
// GNU: declare dso_local ptr @__dynamic_cast({{.*}})
