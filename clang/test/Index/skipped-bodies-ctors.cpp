// RUN: env CINDEXTEST_SKIP_FUNCTION_BODIES=1 c-index-test -test-load-source all %s 2>&1 \
// RUN: | FileCheck --implicit-check-not "error:" %s


template <class T>
struct Foo {
  template <class = int>
  Foo(int &a) : a(a) {
  }

  int &a;
};


int bar = Foo<int>(bar).a + Foo<int>(bar).a;
// CHECK-NOT: error: constructor for 'Foo<int>' must explicitly initialize the reference
