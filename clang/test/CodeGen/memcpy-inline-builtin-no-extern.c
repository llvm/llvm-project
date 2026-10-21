// RUN: %clang_cc1 -triple x86_64 -target-cpu x86-64 -tune-cpu generic -funwind-tables=2 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// Verifies that clang-generated *.inline are flagged as internal and preserve
// function, return, and parameter attributes.

typedef unsigned long size_t;

#define AVAILABLE_EXTERNALLY extern inline __attribute__((always_inline)) __attribute__((gnu_inline))
// Clang recognizes an inline builtin and renames it to memcmp.inline to prevent conflict with builtins.
AVAILABLE_EXTERNALLY int memcmp(const void *a, const void *b, size_t c) {
  return __builtin_memcmp(a, b, c);
}

// CHECK-LABEL: define dso_local i32 @bar(
// CHECK:         call i32 @memcmp.inline(ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})
int bar(const void *a, const void *b, size_t c) {
  return memcmp(a, b, c);
}

// CHECK: declare i32 @memcmp(ptr noundef, ptr noundef, i64 noundef) #[[ATTR_DECL:[0-9]+]]
// CHECK: define internal i32 @memcmp.inline(ptr noundef %a, ptr noundef %b, i64 noundef %c) #[[ATTR_INLINE:[0-9]+]]

// Note that extern has been omitted here.
#define TRIPLE_INLINE inline __attribute__((always_inline)) __attribute__((gnu_inline))

// Clang recognizes an inline builtin and renames it to memcpy.inline to prevent conflict with builtins.
TRIPLE_INLINE __attribute__((returns_nonnull)) void *memcpy(void *a, const void *b, size_t c) {
  return __builtin_memcpy(a, b, c);
}

// CHECK: declare nonnull ptr @memcpy(ptr noundef, ptr noundef, i64 noundef) #[[ATTR_DECL]]
// CHECK: define internal nonnull ptr @memcpy.inline(ptr noundef %a, ptr noundef %b, i64 noundef %c) #[[ATTR_INLINE]]

// CHECK-LABEL: define dso_local ptr @foo(
// CHECK:         call nonnull ptr @memcpy.inline(ptr noundef %{{.*}}, ptr noundef %{{.*}}, i64 noundef %{{.*}})
void *foo(void *a, const void *b, size_t c) {
  return memcpy(a, b, c);
}

// CHECK: attributes #[[ATTR_DECL]] = { alwaysinline nobuiltin nounwind uwtable {{.*}}"target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK: attributes #[[ATTR_INLINE]] = { alwaysinline nobuiltin nounwind uwtable "min-legal-vector-width"="0" {{.*}}"target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
