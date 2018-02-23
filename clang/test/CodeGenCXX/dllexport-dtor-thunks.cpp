// RUN: %clang_cc1 -mconstructor-aliases -fms-extensions %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s

struct __declspec(dllexport) A { virtual ~A(); };
struct __declspec(dllexport) B { virtual ~B(); };
struct __declspec(dllexport) C : A, B { virtual ~C(); };
C::~C() {}

// This thunk should *not* be dllexport.
// CHECK: define linkonce_odr dso_local i8* @"\01??_EC@@W7EAAPEAXI@Z"
// CHECK: define dso_local dllexport void @"\01??1C@@UEAA@XZ"
