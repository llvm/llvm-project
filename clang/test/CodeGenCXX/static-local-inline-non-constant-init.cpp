// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefix=ITANIUM
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++17 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSVC

// A static local in an inline function is shared by every translation unit
// that uses the function. If its initializer isn't a constant initializer,
// another translation unit might not be able to fold it (here, one that can't
// see the initializer of kSize) and will initialize the variable dynamically,
// behind a guard variable. So it must not be emitted as a (possibly read-only)
// constant without a guard in this translation unit either.
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

extern const int kSize = 3;

const int *a() { return &instance(); }
const int *b() { return &constant(); }
const int *c() { return &internal(); }

// ITANIUM-DAG: @_ZZ8instancevE4meta = linkonce_odr global i32 0, comdat, align 4
// ITANIUM-DAG: @_ZGVZ8instancevE4meta = linkonce_odr global i64 0, comdat, align 8
// ITANIUM-DAG: @_ZZ8constantvE1c = linkonce_odr constant i32 42, comdat, align 4
// ITANIUM-DAG: @_ZZL8internalvE1i = internal constant i32 3, align 4
// ITANIUM-NOT: @_ZGVZ8constantvE1c
// ITANIUM-NOT: @_ZGVZL8internalvE1i

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

// MSVC-DAG: @"?meta@?1??instance@@YAAEBHXZ@4HB" = linkonce_odr dso_local global i32 0, comdat, align 4
// MSVC-DAG: @"?$TSS0@?1??instance@@YAAEBHXZ@4HA" = linkonce_odr global i32 0, comdat, align 4
// MSVC-DAG: @"?c@?1??constant@@YAAEBHXZ@4HB" = linkonce_odr dso_local constant i32 42, comdat, align 4
// MSVC-DAG: @"?i@?1??internal@@YAAEBHXZ@4HB" = internal constant i32 3, align 4

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
