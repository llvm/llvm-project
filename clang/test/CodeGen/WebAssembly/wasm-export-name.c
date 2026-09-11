// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

int __attribute__((export_name("bar"))) foo(void);

int foo(void) {
  return 43;
}

int __attribute__((export_name)) default_func(void) {
  return 44;
}

// CHECK: @llvm.used = appending global [2 x ptr] [ptr @foo, ptr @default_func]

// CHECK: define i32 @foo() [[A:#[0-9]+]]
// CHECK: define i32 @default_func() [[B:#[0-9]+]]

// CHECK: attributes [[A]] = {{{.*}} "wasm-export-name"="bar" {{.*}}}
// CHECK: attributes [[B]] = {{{.*}} "wasm-export-name"="default_func" {{.*}}}
