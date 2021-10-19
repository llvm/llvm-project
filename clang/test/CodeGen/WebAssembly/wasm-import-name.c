// RUN: %clang_cc1 -triple wasm32-unknown-unknown-wasm -emit-llvm -o - %s | FileCheck %s

void __attribute__((import_name("bar"))) foo(void);

void call(void) {
  foo();
}

// CHECK: declare void @foo() [[A:#[0-9]+]]

// CHECK: attributes [[A]] = {{{.*}} "wasm-import-name"="bar" {{.*}}}
