// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

struct Simple {
  int x_;
  Simple() {
    x_ = 5;
  }
  ~Simple() {
    x_ += 1;
  }
};

Simple s;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK: define {{.*}}SimpleD2Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback_fields(
// CHECK: ret void

// Destructor does not emit any tail calls
// CHECK: attributes [[ATTRIBUTE]] = {{.*}}"disable-tail-calls"="true"
