// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// CHECK: ModuleID
// CHECK-NOT: zeroinitializer
// CHECK: define i8* @f

struct s {
    int a;
};

static void *v;

static struct s a;

static struct s a = {
    10
};

void *f()
{
  if (a.a)
    return v;
}
