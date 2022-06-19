// REQUIRES: webassembly-registered-target
// RUN: %clang_cc1 -triple wasm32 -emit-llvm %s -o - | FileCheck %s

int foo();

int bar(int a) {
  return foo();
}

int baz() {
  return foo();
}

// CHECK: define i32 @bar(i32 noundef %a) [[BAR_ATTR:#[0-9]+]] {
// CHECK: declare i32 @foo(...) [[FOO_ATTR:#[0-9]+]]
// CHECK: define i32 @baz() [[BAZ_ATTR:#[0-9]+]] {

// CHECK: attributes [[FOO_ATTR]] = {  {{.*}}"no-prototype"{{.*}} }
// CHECK-NOT: attributes [[BAR_ATTR]] = {  {{.*}}"no-prototype"{{.*}} }
// CHECK-NOT: attributes [[BAZ_ATTR]] = {  {{.*}}"no-prototype"{{.*}} }
