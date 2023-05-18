// RUN: env CINDEXTEST_SKIP_FUNCTION_BODIES=1 c-index-test -test-load-source all %s 2>&1 \
// RUN: | FileCheck %s


template <class T>
struct Foo {
  inline int with_body() {
    return 100;
  }

  inline int without_body();
};


int bar = Foo<int>().with_body() + Foo<int>().without_body();
// CHECK-NOT: warning: inline function 'Foo<int>::with_body' is not defined
// CHECK: warning: inline function 'Foo<int>::without_body' is not defined

template <class T>
inline int with_body() { return 10; }

template <class T>
inline int without_body();

int baz = with_body<int>() + without_body<int>();
// CHECK-NOT: warning: inline function 'with_body<int>' is not defined
// CHECK: warning: inline function 'without_body<int>' is not defined
