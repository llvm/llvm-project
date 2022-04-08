// RUN: %clang_cc1 -no-opaque-pointers -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

int __attribute__((export_name("bar"))) foo(void);

int foo(void) {
  return 43;
}

// CHECK: @llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @foo to i8*)]

// CHECK: define i32 @foo() [[A:#[0-9]+]]

// CHECK: attributes [[A]] = {{{.*}} "wasm-export-name"="bar" {{.*}}}
