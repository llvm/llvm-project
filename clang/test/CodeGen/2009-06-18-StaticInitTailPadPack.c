// RUN: %clang_cc1 %s -emit-llvm -o -

  typedef struct A *Foo;
#pragma pack(push, 2)
  struct Bar {
    Foo             f1;
    unsigned short  f2;
    float           f3;
  };
  struct Baz {
    struct Bar   f1;
    struct Bar   f2;
  };
  struct Qux {
    unsigned long   f1;
    struct Baz             f2;
  };
extern const struct Qux Bork;
const struct Qux Bork = {
  0,
  {
    {0},
    {0}
  }
};
