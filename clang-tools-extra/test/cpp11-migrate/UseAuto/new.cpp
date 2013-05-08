// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s

class MyType {
};

class MyDerivedType : public MyType {
};

int main(int argc, char **argv) {
  MyType *a = new MyType();
  // CHECK: auto a = new MyType();

  static MyType *a_static = new MyType();
  // CHECK: static auto a_static = new MyType();

  MyType *b = new MyDerivedType();
  // CHECK: MyType *b = new MyDerivedType();

  void *c = new MyType();
  // CHECK: void *c = new MyType();

  // CV-qualifier tests.
  //
  // NOTE : the form "type const" is expected here because of a deficiency in
  // TypeLoc where CV qualifiers are not considered part of the type location
  // info. That is, all that is being replaced in each case is "MyType *" and
  // not "MyType * const".
  {
    static MyType * const d_static = new MyType();
    // CHECK: static auto const d_static = new MyType();

    MyType * const d3 = new MyType();
    // CHECK: auto const d3 = new MyType();

    MyType * volatile d4 = new MyType();
    // CHECK: auto volatile d4 = new MyType();
  }

  int (**func)(int, int) = new (int(*[5])(int,int));
  // CHECK: int (**func)(int, int) = new (int(*[5])(int,int));

  int *e = new int[5];
  // CHECK: auto e = new int[5];

  MyType *f(new MyType);
  // CHECK: auto f(new MyType);

  MyType *g{new MyType};
  // CHECK: MyType *g{new MyType};
}
