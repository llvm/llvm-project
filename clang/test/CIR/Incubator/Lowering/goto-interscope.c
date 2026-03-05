// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s
struct def;
typedef struct def *decl;
struct def {
  int index;
};
struct def d;
int foo(unsigned char cond, unsigned num)
{
  if (cond)
    goto label;
  {
    decl b = &d;
    label:
      return b->index;
  }

  {
    int a[num];
    if (num > 0)
      return a[0] + a[1];
  }
  return 0;
}
// It is fine enough to check the LLVM IR are generated succesfully.
// CHECK: define {{.*}}i32 @foo
// CHECK: alloca ptr
// CHECK: alloca i8
// Check the dynamic alloca is not hoisted and live in a seperate block.
// CHECK: :
// Check we have a dynamic alloca
// CHECK: alloca i32, i64 %{{.*}}
