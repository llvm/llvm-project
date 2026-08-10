typedef int (^BlockTy)(void);

struct S0 {
  int a;
};

struct S {
  int i;
  void func(BlockTy __attribute__((noescape)));
  int foo(S0 &);

  void m() {
    __block S0 x;
    func(^{ return foo(x); });
  }
};
