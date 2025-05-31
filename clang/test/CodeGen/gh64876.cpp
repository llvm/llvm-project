// RUN: %clang_cc1 -triple x86_64 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

void f(const char* C, const wchar_t *WC) {
  int x1 = __builtin_strncmp(C, "b", 0xffffffffffffffff);
// CHECK: {{.*}}= call i32 @strncmp{{.*}}i64 noundef -1
  int x2 = __builtin_memcmp(C, "b", 0xffffffffffffffff);
// CHECK: {{.*}}= call i32 @memcmp{{.*}}i64 noundef -1
  int x3 = __builtin_bcmp(C, "b", 0xffffffffffffffff);
// CHECK: {{.*}}= call i32 @bcmp{{.*}}i64 noundef -1
  int x4 = __builtin_wmemcmp(WC, L"b", 0xffffffffffffffff);
// CHECK: {{.*}}= call i32 @wmemcmp{{.*}}i64 noundef -1
  auto x5 = __builtin_memchr(C, (int)'a', 0xffffffffffffffff);
// CHECK: {{.*}}= call ptr @memchr{{.*}}i64 noundef -1
  auto x6 = __builtin_wmemchr(WC, (int)'a', 0xffffffffffffffff);
// CHECK: {{.*}}= call ptr @wmemchr{{.*}}i64 noundef -1
}
