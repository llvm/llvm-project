// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o -
// RUN: %clang_cc1 -triple %ms_abi_triple -fno-rtti -emit-llvm %s -o -

struct A {
  virtual void Method() = 0;
};

struct B : public A {
  virtual void Method() { }
};

typedef void (A::*fn_type_a)(void);
typedef void (B::*fn_type_b)(void);

int main(int argc, char **argv)
{
  fn_type_a f = reinterpret_cast<fn_type_a>(&B::Method);
  fn_type_b g = reinterpret_cast<fn_type_b>(f);
  B b;
  (b.*g)();
  return 0;
}
