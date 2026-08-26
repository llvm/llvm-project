// Hexagon lowers KCFI operand bundles in the back end. Clang must leave the "kcfi"
// operand bundles in place for the back end instead of running the middle-end
// KCFIPass, which would rewrite them into a software llvm.debugtrap check.
//
// Verify the bundle survives the optimizer pipeline at both -O0 and -O2 and
// is not lowered to debugtrap.
//
// RUN: %clang_cc1 -triple hexagon-unknown-linux-musl -O0 -fsanitize=kcfi -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple hexagon-unknown-linux-musl -O2 -fsanitize=kcfi -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define {{.*}}void @call(
// CHECK: call void %{{.*}}() {{.*}}[ "kcfi"(i32 {{-?[0-9]+}}) ]
// CHECK-NOT: @llvm.debugtrap
void call(void (*f)(void)) { f(); }
