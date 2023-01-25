// RUN: %clang_cc1 -emit-llvm %s -o - | grep llvm.global.annotations
// RUN: %clang_cc1 -emit-llvm %s -o - | grep llvm.var.annotation | count 3

/* Global variable with attribute */
int X __attribute__((annotate("GlobalValAnnotation")));

/* Function with attribute */
int foo(int y) __attribute__((annotate("GlobalValAnnotation")))
               __attribute__((noinline));

int foo(int y __attribute__((annotate("LocalValAnnotation")))) {
  int x __attribute__((annotate("LocalValAnnotation")));
  x = 34;
  return y + x;
}

/* Attribute with struct argument. */
struct TestStruct {
  int a;
  int b;
};
int Y __attribute__((annotate(
  "GlobalValAnnotationWithArgs", 
  42,
  (struct TestStruct) { .a = 1, .b = 2 }
)));

// CHECK: @.str.3 = private unnamed_addr constant [28 x i8] c"GlobalValAnnotationWithArgs\00", section "llvm.metadata"
// CHECK-NEXT: @.args = private unnamed_addr constant { i32, %struct.TestStruct } { i32 42, %struct.TestStruct { i32 1, i32 2 } }, section "llvm.metadata"

int main(void) {
  static int a __attribute__((annotate("GlobalValAnnotation")));
  a = foo(2);
  return 0;
}
