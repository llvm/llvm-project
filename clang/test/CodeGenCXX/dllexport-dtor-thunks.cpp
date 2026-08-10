// RUN: %clang_cc1 -mconstructor-aliases -fms-extensions %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s

namespace test1 {
struct A { ~A(); };
struct __declspec(dllexport) B : virtual A { };
// CHECK: define weak_odr dso_local dllexport void @"??1B@test1@@QEAA@XZ"
// CHECK: define weak_odr dso_local dllexport void @"??_DB@test1@@QEAAXXZ"
}

struct __declspec(dllexport) A { virtual ~A(); };
struct __declspec(dllexport) B { virtual ~B(); };
struct __declspec(dllexport) C : A, B { virtual ~C(); };
C::~C() {}

// CHECK: define dso_local dllexport void @"??1C@@UEAA@XZ"
// This thunk should *not* be dllexport.
// CHECK: define linkonce_odr dso_local noundef ptr @"??_EC@@W7EAAPEAXI@Z"
