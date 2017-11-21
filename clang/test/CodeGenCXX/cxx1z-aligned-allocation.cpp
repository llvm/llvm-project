// Check that delete exprs call aligned (de)allocation functions if
// -faligned-allocation is passed in both C++11 and C++14.
// RUN: %clang_cc1 -std=c++11 -fexceptions -fsized-deallocation -faligned-allocation %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -fexceptions -fsized-deallocation -faligned-allocation %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++1z -fexceptions -fsized-deallocation %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s

// RUN: %clang_cc1 -std=c++1z -fexceptions -fsized-deallocation %s -emit-llvm -triple x86_64-windows-msvc -o - | FileCheck %s --check-prefix=CHECK-MS

// Check that we don't used aligned (de)allocation without -faligned-allocation or C++1z.
// RUN: %clang_cc1 -std=c++14 -DUNALIGNED -fexceptions %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefix=CHECK-UNALIGNED
// RUN: %clang_cc1 -std=c++1z -DUNALIGNED -fexceptions -fno-aligned-allocation %s -emit-llvm -triple x86_64-linux-gnu -o - | FileCheck %s --check-prefix=CHECK-UNALIGNED

// CHECK-UNALIGNED-NOT: _Znwm_St11align_val_t
// CHECK-UNALIGNED-NOT: _Znam_St11align_val_t
// CHECK-UNALIGNED-NOT: _ZdlPv_St11align_val_t
// CHECK-UNALIGNED-NOT: _ZdaPv_St11align_val_t
// CHECK-UNALIGNED-NOT: _ZdlPvm_St11align_val_t
// CHECK-UNALIGNED-NOT: _ZdaPvm_St11align_val_t

typedef decltype(sizeof(0)) size_t;
namespace std { enum class align_val_t : size_t {}; }

#define OVERALIGNED alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2)

// Global new and delete.
// ======================
struct OVERALIGNED A { A(); int n[128]; };

// CHECK-LABEL: define {{.*}} @_Z2a0v()
// CHECK: %[[ALLOC:.*]] = call i8* @_ZnwmSt11align_val_t(i64 512, i64 32)
// CHECK: call void @_ZdlPvSt11align_val_t(i8* %[[ALLOC]], i64 32)
// CHECK-MS-LABEL: define {{.*}} @"\01?a0@@YAPEAXXZ"()
// CHECK-MS: %[[ALLOC:.*]] = call i8* @"\01??2@YAPEAX_KW4align_val_t@std@@@Z"(i64 512, i64 32)
// CHECK-MS: cleanuppad
// CHECK-MS: call void @"\01??3@YAXPEAXW4align_val_t@std@@@Z"(i8* %[[ALLOC]], i64 32)
void *a0() { return new A; }

// FIXME: Why don't we call the sized array deallocation overload in this case?
// The size is known.
//
// CHECK-LABEL: define {{.*}} @_Z2a1l(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZnamSt11align_val_t(i64 %{{.*}}, i64 32)
// No array cookie.
// CHECK-NOT: store
// CHECK: invoke void @_ZN1AC1Ev(
// CHECK: call void @_ZdaPvSt11align_val_t(i8* %[[ALLOC]], i64 32)
// CHECK-MS-LABEL: define {{.*}} @"\01?a1@@YAPEAXJ@Z"(
// CHECK-MS: %[[ALLOC:.*]] = call i8* @"\01??_U@YAPEAX_KW4align_val_t@std@@@Z"(i64 %{{.*}}, i64 32)
// No array cookie.
// CHECK-MS-NOT: store
// CHECK-MS: invoke %struct.A* @"\01??0A@@QEAA@XZ"(
// CHECK-MS: cleanuppad
// CHECK-MS: call void @"\01??_V@YAXPEAXW4align_val_t@std@@@Z"(i8* %[[ALLOC]], i64 32)
void *a1(long n) { return new A[n]; }

// CHECK-LABEL: define {{.*}} @_Z2a2P1A(
// CHECK: call void @_ZdlPvmSt11align_val_t(i8* %{{.*}}, i64 512, i64 32) #9
void a2(A *p) { delete p; }

// CHECK-LABEL: define {{.*}} @_Z2a3P1A(
// CHECK: call void @_ZdaPvSt11align_val_t(i8* %{{.*}}, i64 32) #9
void a3(A *p) { delete[] p; }


// Class-specific usual new and delete.
// ====================================
struct OVERALIGNED B {
  B();
  // These are just a distraction. We should ignore them.
  void *operator new(size_t);
  void operator delete(void*, size_t);
  void operator delete[](void*, size_t);

  void *operator new(size_t, std::align_val_t);
  void operator delete(void*, std::align_val_t);
  void operator delete[](void*, std::align_val_t);

  int n[128];
};

// CHECK-LABEL: define {{.*}} @_Z2b0v()
// CHECK: %[[ALLOC:.*]] = call i8* @_ZN1BnwEmSt11align_val_t(i64 512, i64 32)
// CHECK: call void @_ZN1BdlEPvSt11align_val_t(i8* %[[ALLOC]], i64 32)
void *b0() { return new B; }

// CHECK-LABEL: define {{.*}} @_Z2b1l(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZnamSt11align_val_t(i64 %{{.*}}, i64 32)
// No array cookie.
// CHECK-NOT: store
// CHECK: invoke void @_ZN1BC1Ev(
// CHECK: call void @_ZN1BdaEPvSt11align_val_t(i8* %[[ALLOC]], i64 32)
void *b1(long n) { return new B[n]; }

// CHECK-LABEL: define {{.*}} @_Z2b2P1B(
// CHECK: call void @_ZN1BdlEPvSt11align_val_t(i8* %{{.*}}, i64 32)
void b2(B *p) { delete p; }

// CHECK-LABEL: define {{.*}} @_Z2b3P1B(
// CHECK: call void @_ZN1BdaEPvSt11align_val_t(i8* %{{.*}}, i64 32)
void b3(B *p) { delete[] p; }

struct OVERALIGNED C {
  C();
  void *operator new[](size_t, std::align_val_t);
  void operator delete[](void*, size_t, std::align_val_t);

  // It doesn't matter that we have an unaligned operator delete[] that doesn't
  // want the size. What matters is that the aligned one does.
  void operator delete[](void*);
};

// This one has an array cookie.
// CHECK-LABEL: define {{.*}} @_Z2b4l(
// CHECK: call {{.*}} @llvm.umul.with.overflow{{.*}}i64 32
// CHECK: call {{.*}} @llvm.uadd.with.overflow{{.*}}i64 32
// CHECK: %[[ALLOC:.*]] = call i8* @_ZN1CnaEmSt11align_val_t(i64 %{{.*}}, i64 32)
// CHECK: store
// CHECK: call void @_ZN1CC1Ev(
//
// Note, we're still calling a placement allocation function, and there is no
// matching placement operator delete. =(
// FIXME: This seems broken.
// CHECK-NOT: call void @_ZN1CdaEPvmSt11align_val_t(
#ifndef UNALIGNED
void *b4(long n) { return new C[n]; }
#endif

// CHECK-LABEL: define {{.*}} @_Z2b5P1C(
// CHECK: mul i64{{.*}} 32
// CHECK: add i64{{.*}} 32
// CHECK: call void @_ZN1CdaEPvmSt11align_val_t(
void b5(C *p) { delete[] p; }


// Global placement new.
// =====================

struct Q { int n; } q;
void *operator new(size_t, Q);
void *operator new(size_t, std::align_val_t, Q);
void operator delete(void*, Q);
void operator delete(void*, std::align_val_t, Q);

// CHECK-LABEL: define {{.*}} @_Z2c0v(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZnwmSt11align_val_t1Q(i64 512, i64 32, i32 %
// CHECK: call void @_ZdlPvSt11align_val_t1Q(i8* %[[ALLOC]], i64 32, i32 %
void *c0() { return new (q) A; }


// Class-specific placement new.
// =============================

struct OVERALIGNED D {
  D();
  void *operator new(size_t, Q);
  void *operator new(size_t, std::align_val_t, Q);
  void operator delete(void*, Q);
  void operator delete(void*, std::align_val_t, Q);
};

// CHECK-LABEL: define {{.*}} @_Z2d0v(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZN1DnwEmSt11align_val_t1Q(i64 32, i64 32, i32 %
// CHECK: call void @_ZN1DdlEPvSt11align_val_t1Q(i8* %[[ALLOC]], i64 32, i32 %
void *d0() { return new (q) D; }


// Calling aligned new with placement syntax.
// ==========================================

#ifndef UNALIGNED
// CHECK-LABEL: define {{.*}} @_Z2e0v(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZnwmSt11align_val_t(i64 512, i64 5)
// CHECK: call void @_ZdlPvSt11align_val_t(i8* %[[ALLOC]], i64 5)
void *e0() { return new (std::align_val_t(5)) A; }

// CHECK-LABEL: define {{.*}} @_Z2e1v(
// CHECK: %[[ALLOC:.*]] = call i8* @_ZN1BnwEmSt11align_val_t(i64 512, i64 5)
// CHECK: call void @_ZN1BdlEPvSt11align_val_t(i8* %[[ALLOC]], i64 5)
void *e1() { return new (std::align_val_t(5)) B; }
#endif

// Variadic placement/non-placement allocation functions.
// ======================================================

struct OVERALIGNED F {
  F();
  void *operator new(size_t, ...);
  void operator delete(void*, ...);
  int n[128];
};

// CHECK-LABEL: define {{.*}} @_Z2f0v(
// CHECK: %[[ALLOC:.*]] = call i8* (i64, ...) @_ZN1FnwEmz(i64 512, i64 32)
// Non-placement allocation function, uses normal deallocation lookup which
// cares about whether a parameter has type std::align_val_t.
// CHECK: call void (i8*, ...) @_ZN1FdlEPvz(i8* %[[ALLOC]])
void *f0() { return new F; }

// CHECK-LABEL: define {{.*}} @_Z2f1v(
// CHECK: %[[ALLOC:.*]] = call i8* (i64, ...) @_ZN1FnwEmz(i64 512, i64 32, i32 %
// Placement allocation function, uses placement deallocation matching, which
// passes same arguments and therefore includes alignment.
// CHECK: call void (i8*, ...) @_ZN1FdlEPvz(i8* %[[ALLOC]], i64 32, i32 %
void *f1() { return new (q) F; }

struct OVERALIGNED G {
  G();
  void *operator new(size_t, std::align_val_t, ...);
  void operator delete(void*, std::align_val_t, ...);
  int n[128];
};
#ifndef UNALIGNED
// CHECK-LABEL: define {{.*}} @_Z2g0v
// CHECK: %[[ALLOC:.*]] = call i8* (i64, i64, ...) @_ZN1GnwEmSt11align_val_tz(i64 512, i64 32)
// CHECK: call void (i8*, i64, ...) @_ZN1GdlEPvSt11align_val_tz(i8* %[[ALLOC]], i64 32)
void *g0() { return new G; }

// CHECK-LABEL: define {{.*}} @_Z2g1v
// CHECK: %[[ALLOC:.*]] = call i8* (i64, i64, ...) @_ZN1GnwEmSt11align_val_tz(i64 512, i64 32, i32 %
// CHECK: call void (i8*, i64, ...) @_ZN1GdlEPvSt11align_val_tz(i8* %[[ALLOC]], i64 32, i32 %
void *g1() { return new (q) G; }
#endif
