// RUN: %clang_cc1 %s -S -emit-llvm -triple x86_64-unknown-linux-gnu -o - -Wno-c++23-extensions | FileCheck %s

[[clang::builtin("memcpy")]] void* my_memcpy(void*, const void*, unsigned long);

void call_memcpy(int i) {
  int j;
  my_memcpy(&j, &i, sizeof(int));

  // CHECK:      define dso_local void @_Z11call_memcpyi(i32 noundef %i) #0 {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %i.addr = alloca i32, align 4
  // CHECK-NEXT:   %j = alloca i32, align 4
  // CHECK-NEXT:   store i32 %i, ptr %i.addr, align 4
  // CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %j, ptr align 4 %i.addr, i64 4, i1 false)
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
}

[[clang::builtin("memmove")]] char* typed_memmove(char*, const char*, unsigned long);

char* call_typed_memmove(char* dst, const char* src, unsigned long count) {
  return typed_memmove(dst, src, count);

  // CHECK:      define dso_local noundef ptr @_Z18call_typed_memmovePcPKcm(ptr noundef %dst, ptr noundef %src, i64 noundef %count) #0 {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %dst.addr = alloca ptr, align 8
  // CHECK-NEXT:   %src.addr = alloca ptr, align 8
  // CHECK-NEXT:   %count.addr = alloca i64, align 8
  // CHECK-NEXT:   store ptr %dst, ptr %dst.addr, align 8
  // CHECK-NEXT:   store ptr %src, ptr %src.addr, align 8
  // CHECK-NEXT:   store i64 %count, ptr %count.addr, align 8
  // CHECK-NEXT:   %0 = load ptr, ptr %dst.addr, align 8
  // CHECK-NEXT:   %1 = load ptr, ptr %src.addr, align 8
  // CHECK-NEXT:   %2 = load i64, ptr %count.addr, align 8
  // CHECK-NEXT:   call void @llvm.memmove.p0.p0.i64(ptr align 1 %0, ptr align 1 %1, i64 %2, i1 false)
  // CHECK-NEXT:   ret ptr %0
  // CHECK-NEXT: }
}

template <class T>
[[clang::builtin("std::move")]] __remove_reference_t(T)&& my_move(T&&);

void call_move() {
  int i = my_move(0);

  // CHECK:      define dso_local void @_Z9call_movev() #0 {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %i = alloca i32, align 4
  // CHECK-NEXT:   %ref.tmp = alloca i32, align 4
  // CHECK-NEXT:   store i32 0, ptr %ref.tmp, align 4
  // CHECK-NEXT:   %0 = load i32, ptr %ref.tmp, align 4
  // CHECK-NEXT:   store i32 %0, ptr %i, align 4
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
}

struct identity {
  template <class T>
  [[clang::builtin("std::forward")]] static T&& operator()(T&&) noexcept;
};

void call_identity() {
  (void)identity()(1);

  // CHECK:      define dso_local void @_Z13call_identityv() #0 {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ref.tmp = alloca i32, align 4
  // CHECK-NEXT:   store i32 1, ptr %ref.tmp, align 4
  // CHECK-NEXT:   ret void
  // CHECK-NEXT: }
}
