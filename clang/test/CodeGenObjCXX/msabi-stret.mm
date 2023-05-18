// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fobjc-runtime=ios-6.0 -Os -S -emit-llvm -o - %s -mframe-pointer=all | FileCheck %s

struct S {
  S() = default;
  S(const S &) {}
};

@interface I
+ (S)m:(S)s;
@end

S f() {
  return [I m:S()];
}

// CHECK: declare dso_local void @objc_msgSend_stret(ptr, ptr, ...)
// CHECK-NOT: declare dllimport void @objc_msgSend(ptr, ptr, ...)
