// Test to check that the appropriate attributes are added to the __hot_cold_t
// versions of operator new.

// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -faligned-allocation -emit-llvm -o - | FileCheck %s

typedef __typeof__(sizeof(0)) size_t;

namespace std {
struct nothrow_t {};
enum class align_val_t : size_t;
} // namespace std
std::nothrow_t nothrow;

typedef unsigned char uint8_t;

// See the following link for how this type is declared in tcmalloc:
// https://github.com/google/tcmalloc/blob/220043886d4e2efff7a5702d5172cb8065253664/tcmalloc/malloc_extension.h#L53.
enum class __hot_cold_t : uint8_t;

// Test handling of declaration that uses a type alias, ensuring that it still
// recognizes the expected __hot_cold_t type name.
namespace malloc_namespace {
using hot_cold_t = __hot_cold_t;
}

void *operator new(size_t size,
                   malloc_namespace::hot_cold_t hot_cold) noexcept(false);
void *operator new[](size_t size,
                     malloc_namespace::hot_cold_t hot_cold) noexcept(false);
void *operator new(size_t size, const std::nothrow_t &,
                   malloc_namespace::hot_cold_t hot_cold) noexcept;
void *operator new[](size_t size, const std::nothrow_t &,
                     malloc_namespace::hot_cold_t hot_cold) noexcept;
void *operator new(size_t size, std::align_val_t alignment,
                   malloc_namespace::hot_cold_t hot_cold) noexcept(false);
void *operator new[](size_t size, std::align_val_t alignment,
                     malloc_namespace::hot_cold_t hot_cold) noexcept(false);
void *operator new(size_t size, std::align_val_t alignment,
                   const std::nothrow_t &,
                   malloc_namespace::hot_cold_t hot_cold) noexcept;
void *operator new[](size_t size, std::align_val_t alignment,
                     const std::nothrow_t &,
                     malloc_namespace::hot_cold_t hot_cold) noexcept;

// All explicit operator new calls should not get any builtin attribute, whereas
// all implicit new expressions should get builtin attributes. All of the
// declarations should get nobuiltin attributes.

void hot_cold_new() {
  // CHECK: call noalias noundef nonnull ptr @_Znwm12__hot_cold_t(i64 noundef 1, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_CALL:#[^ ]*]]
  operator new(1, (__hot_cold_t)0);
  // CHECK: call noalias noundef nonnull ptr @_Znwm12__hot_cold_t(i64 noundef 4, i8 noundef zeroext 0) [[ATTR_BUILTIN_CALL:#[^ ]*]]
  new ((__hot_cold_t)0) int;
}

// CHECK: declare noundef nonnull ptr @_Znwm12__hot_cold_t(i64 noundef, i8 noundef zeroext) [[ATTR_NOBUILTIN:#[^ ]*]]

void hot_cold_new_array() {
  // CHECK: call noalias noundef nonnull ptr @_Znam12__hot_cold_t(i64 noundef 1, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_CALL:#[^ ]*]]
  operator new[](1, (__hot_cold_t)0);
  // CHECK: call noalias noundef nonnull ptr @_Znam12__hot_cold_t(i64 noundef 4, i8 noundef zeroext 0) [[ATTR_BUILTIN_CALL:#[^ ]*]]
  new ((__hot_cold_t)0) int[1];
}

// CHECK: declare noundef nonnull ptr @_Znam12__hot_cold_t(i64 noundef, i8 noundef zeroext) [[ATTR_NOBUILTIN:#[^ ]*]]

void hot_cold_new_nothrow() {
  // CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 noundef 1, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  operator new(1, nothrow, (__hot_cold_t)0);
  // CHECK: call noalias noundef ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 noundef 4, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  new (nothrow, (__hot_cold_t)0) int;
}

// CHECK: declare noundef ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64 noundef, ptr noundef nonnull align 1 dereferenceable(1), i8 noundef zeroext) [[ATTR_NOBUILTIN_NOTHROW:#[^ ]*]]

void hot_cold_new_nothrow_array() {
  // CHECK: call noalias noundef ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 noundef 1, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  operator new[](1, nothrow, (__hot_cold_t)0);
  // CHECK: call noalias noundef ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 noundef 4, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  new (nothrow, (__hot_cold_t)0) int[1];
}

// CHECK: declare noundef ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64 noundef, ptr noundef nonnull align 1 dereferenceable(1), i8 noundef zeroext) [[ATTR_NOBUILTIN_NOTHROW:#[^ ]*]]

class alignas(32) alignedstruct {
  int x;
};

void hot_cold_new_align() {
  // CHECK: call noalias noundef nonnull align 32 ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 noundef 1, i64 noundef 32, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_CALL:#[^ ]*]]
  operator new(1, (std::align_val_t)32, (__hot_cold_t)0);
  // CHECK: call noalias noundef nonnull align 32 ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 noundef 32, i64 noundef 32, i8 noundef zeroext 0) [[ATTR_BUILTIN_CALL:#[^ ]*]]
  new ((__hot_cold_t)0) alignedstruct;
}

// CHECK: declare noundef nonnull ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64 noundef, i64 noundef, i8 noundef zeroext) [[ATTR_NOBUILTIN:#[^ ]*]]

void hot_cold_new_align_array() {
  // CHECK: call noalias noundef nonnull align 32 ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 noundef 1, i64 noundef 32, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_CALL:#[^ ]*]]
  operator new[](1, (std::align_val_t)32, (__hot_cold_t)0);
  // CHECK: call noalias noundef nonnull align 32 ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 noundef 32, i64 noundef 32, i8 noundef zeroext 0) [[ATTR_BUILTIN_CALL:#[^ ]*]]
  new ((__hot_cold_t)0) alignedstruct[1];
}

// CHECK: declare noundef nonnull ptr @_ZnamSt11align_val_t12__hot_cold_t(i64 noundef, i64 noundef, i8 noundef zeroext) [[ATTR_NOBUILTIN:#[^ ]*]]

void hot_cold_new_align_nothrow() {
  // CHECK: call noalias noundef align 32 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef 1, i64 noundef 32, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  operator new(1, (std::align_val_t)32, nothrow, (__hot_cold_t)0);
  // CHECK: call noalias noundef align 32 ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef 32, i64 noundef 32, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  new (nothrow, (__hot_cold_t)0) alignedstruct;
}

// CHECK: declare noundef ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef, i64 noundef, ptr noundef nonnull align 1 dereferenceable(1), i8 noundef zeroext) [[ATTR_NOBUILTIN_NOTHROW:#[^ ]*]]

void hot_cold_new_align_nothrow_array() {
  // CHECK: call noalias noundef align 32 ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef 1, i64 noundef 32, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_NO_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  operator new[](1, (std::align_val_t)32, nothrow, (__hot_cold_t)0);
  // CHECK: call noalias noundef align 32 ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef 32, i64 noundef 32, {{.*}} @nothrow, i8 noundef zeroext 0) [[ATTR_BUILTIN_NOTHROW_CALL:#[^ ]*]]
  new (nothrow, (__hot_cold_t)0) alignedstruct[1];
}

// CHECK: declare noundef ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64 noundef, i64 noundef, ptr noundef nonnull align 1 dereferenceable(1), i8 noundef zeroext) [[ATTR_NOBUILTIN_NOTHROW:#[^ ]*]]

// CHECK-DAG: attributes [[ATTR_NOBUILTIN]] = { nobuiltin allocsize(0) {{.*}} }
// CHECK-DAG: attributes [[ATTR_NOBUILTIN_NOTHROW]] = { nobuiltin nounwind allocsize(0) {{.*}} }
// CHECK-DAG: attributes [[ATTR_NO_BUILTIN_CALL]] = { allocsize(0) }
// CHECK-DAG: attributes [[ATTR_BUILTIN_CALL]] = { builtin allocsize(0) }
// CHECK-DAG: attributes [[ATTR_NO_BUILTIN_NOTHROW_CALL]] = { nounwind allocsize(0) }
// CHECK-DAG: attributes [[ATTR_BUILTIN_NOTHROW_CALL]] = { builtin nounwind allocsize(0) }
