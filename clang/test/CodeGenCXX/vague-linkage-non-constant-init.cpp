// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC

// A variable with vague linkage (a static local of an inline function, an
// inline variable, a static data member of a class template) is shared by
// every translation unit that uses it. If its initializer isn't a constant
// initializer, another translation unit might not be able to fold it (here, one
// that can't see the initializer of kSize) and will initialize the variable
// dynamically. So it must not be emitted as a (possibly read-only) constant
// without a guard in this translation unit either.
// See https://github.com/llvm/llvm-project/issues/226631.

extern const int kSize;

inline const int &instance() {
  static const int meta = kSize;
  return meta;
}

// Constant initializers are still emitted as constants, without a guard.
inline const int &constant() {
  static const int c = 42;
  return c;
}

// A static local that isn't shared with other translation units can still be
// folded, even if its initializer isn't a constant initializer.
static const int &internal() {
  static const int i = kSize;
  return i;
}

inline const int iv = kSize;
inline int mv = kSize;
inline const int ic = 42;

template <class T> struct S { static const int x; };
template <class T> const int S<T>::x = kSize;

static const int sv = kSize;

const int *d() { return &iv; }
int *e() { return &mv; }
const int *f() { return &ic; }
const int *g() { return &S<int>::x; }
const int *h() { return &sv; }

extern const int kSize = 3;

const int *a() { return &instance(); }
const int *b() { return &constant(); }
const int *c() { return &internal(); }

// ITANIUM-DAG: @_ZZ8instancevE4meta = linkonce_odr global i32 0, comdat, align 4
// ITANIUM-DAG: @_ZGVZ8instancevE4meta = linkonce_odr global i64 0, comdat, align 8
// ITANIUM-DAG: @_ZZ8constantvE1c = linkonce_odr constant i32 42, comdat, align 4
// ITANIUM-DAG: @_ZZL8internalvE1i = internal constant i32 3, align 4
// ITANIUM-DAG: @iv = linkonce_odr global i32 0, comdat, align 4
// ITANIUM-DAG: @_ZGV2iv = linkonce_odr global i64 0, comdat($iv), align 8
// ITANIUM-DAG: @mv = linkonce_odr global i32 0, comdat, align 4
// ITANIUM-DAG: @_ZGV2mv = linkonce_odr global i64 0, comdat($mv), align 8
// ITANIUM-DAG: @ic = linkonce_odr constant i32 42, comdat, align 4
// ITANIUM-DAG: @_ZN1SIiE1xE = linkonce_odr global i32 0, comdat, align 4
// ITANIUM-DAG: @_ZGVN1SIiE1xE = linkonce_odr global i64 0, comdat($_ZN1SIiE1xE), align 8
// ITANIUM-DAG: @_ZL2sv = internal constant i32 3, align 4
// ITANIUM-NOT: @_ZGVZ8constantvE1c
// ITANIUM-NOT: @_ZGVZL8internalvE1i
// ITANIUM-NOT: @_ZGV2ic

// ITANIUM-LABEL: define linkonce_odr {{.*}} ptr @_Z8instancev()
// ITANIUM:         load atomic i8, ptr @_ZGVZ8instancevE4meta acquire
// ITANIUM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ8instancevE4meta)
// ITANIUM:         store i32 3, ptr @_ZZ8instancevE4meta
// ITANIUM:         call void @__cxa_guard_release(ptr @_ZGVZ8instancevE4meta)
// ITANIUM:         ret ptr @_ZZ8instancevE4meta

// ITANIUM-LABEL: define linkonce_odr {{.*}} ptr @_Z8constantv()
// ITANIUM-NEXT:  entry:
// ITANIUM-NEXT:    ret ptr @_ZZ8constantvE1c

// ITANIUM-LABEL: define internal {{.*}} ptr @_ZL8internalv()
// ITANIUM-NEXT:  entry:
// ITANIUM-NEXT:    ret ptr @_ZZL8internalvE1i

// ITANIUM-LABEL: define internal void @__cxx_global_var_init() {{.*}} comdat($iv)
// ITANIUM:         call i32 @__cxa_guard_acquire(ptr @_ZGV2iv)
// ITANIUM:         store i32 3, ptr @iv
// ITANIUM:         call void @__cxa_guard_release(ptr @_ZGV2iv)

// ITANIUM-LABEL: define internal void @__cxx_global_var_init.1() {{.*}} comdat($mv)
// ITANIUM:         call i32 @__cxa_guard_acquire(ptr @_ZGV2mv)
// ITANIUM:         store i32 3, ptr @mv
// ITANIUM:         call void @__cxa_guard_release(ptr @_ZGV2mv)

// ITANIUM-LABEL: define internal void @__cxx_global_var_init.2() {{.*}} comdat($_ZN1SIiE1xE)
// ITANIUM:         load i8, ptr @_ZGVN1SIiE1xE
// ITANIUM:         store i8 1, ptr @_ZGVN1SIiE1xE
// ITANIUM:         store i32 3, ptr @_ZN1SIiE1xE

// MSVC-DAG: @"?meta@?1??instance@@YAAEBHXZ@4HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// MSVC-DAG: @"?$TSS0@?1??instance@@YAAEBHXZ@4HA" = linkonce_odr global i32 0, comdat, align 4
// MSVC-DAG: @"?c@?1??constant@@YAAEBHXZ@4HB" = linkonce_odr dso_local constant i32 42, comdat, align 4
// MSVC-DAG: @"?i@?1??internal@@YAAEBHXZ@4HB" = internal constant i32 3, align 4
// MSVC-DAG: @"?iv@@3HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// MSVC-DAG: @"?mv@@3HA" = linkonce_odr dso_local global i32 0, comdat, align 4
// MSVC-DAG: @"?ic@@3HB" = linkonce_odr dso_local constant i32 42, comdat, align 4
// MSVC-DAG: @"?x@?$S@H@@2HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// MSVC-DAG: @sv = internal constant i32 3, align 4
// MSVC-DAG: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @"??__Eiv@@YAXXZ", ptr @"?iv@@3HB" }, { i32, ptr, ptr } { i32 65535, ptr @"??__Emv@@YAXXZ", ptr @"?mv@@3HA" }, { i32, ptr, ptr } { i32 65535, ptr @"??__E?x@?$S@H@@2HB@@YAXXZ", ptr @"?x@?$S@H@@2HB" }]

// MSVC-LABEL: define linkonce_odr {{.*}} ptr @"?instance@@YAAEBHXZ"()
// MSVC:         call void @_Init_thread_header(ptr @"?$TSS0@?1??instance@@YAAEBHXZ@4HA")
// MSVC:         store i32 3, ptr @"?meta@?1??instance@@YAAEBHXZ@4HB"
// MSVC:         call void @_Init_thread_footer(ptr @"?$TSS0@?1??instance@@YAAEBHXZ@4HA")
// MSVC:         ret ptr @"?meta@?1??instance@@YAAEBHXZ@4HB"

// MSVC-LABEL: define linkonce_odr {{.*}} ptr @"?constant@@YAAEBHXZ"()
// MSVC-NEXT:  entry:
// MSVC-NEXT:    ret ptr @"?c@?1??constant@@YAAEBHXZ@4HB"

// MSVC-LABEL: define internal {{.*}} ptr @"?internal@@YAAEBHXZ"()
// MSVC-NEXT:  entry:
// MSVC-NEXT:    ret ptr @"?i@?1??internal@@YAAEBHXZ@4HB"

// MSVC-LABEL: define linkonce_odr dso_local void @"??__Eiv@@YAXXZ"()
// MSVC:         store i32 3, ptr @"?iv@@3HB"

// MSVC-LABEL: define linkonce_odr dso_local void @"??__Emv@@YAXXZ"()
// MSVC:         store i32 3, ptr @"?mv@@3HA"

// MSVC-LABEL: define linkonce_odr dso_local void @"??__E?x@?$S@H@@2HB@@YAXXZ"()
// MSVC:         store i32 3, ptr @"?x@?$S@H@@2HB"
