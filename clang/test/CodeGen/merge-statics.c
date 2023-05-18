// RUN: %clang_cc1 < %s -emit-llvm | FileCheck %s

// The two decls for 'a' should merge into one llvm GlobalVariable.

struct s { int x; };
static struct s a;

struct s *ap1 = &a;

static struct s a =  {
    10
};

// CHECK-NOT: internal global
// CHECK: @a = internal global %struct.s { i32 10 }
// CHECK-NOT: internal-global
