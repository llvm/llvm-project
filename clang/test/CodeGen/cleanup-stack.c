// RUN: %clang_cc1 -triple i386-unknown-unknown -O3 -emit-llvm %s -o %t
// RUN: grep "ret i32 9" %t

struct s0 {
  int *var;
  int addend;
};

static void f0(struct s0 *p) {
  *p->var += p->addend;
}

int f1(void) {
  int var = 0;
  
  {
    struct s0 x __attribute__((cleanup(f0))) = { &var, 2 };  
    struct s0 y __attribute__((cleanup(f0))) = { &var, 3 };
    {
      struct s0 y __attribute__((cleanup(f0))) = { &var, 4 };
    }
  }

  return var;
}
