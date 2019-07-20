//RUN: %clang_cc1 %s -cl-std=c++ -pedantic -ast-dump -verify | FileCheck %s

//expected-no-diagnostics

//CHECK: |-VarDecl {{.*}} foo 'const __global int'
constexpr int foo = 0;

class c {
public:
  //CHECK: `-VarDecl {{.*}} foo2 'const __global int'
  static constexpr int foo2 = 0;
};

struct c1 {};

// We only deduce addr space in type alias in pointer types.
//CHECK: TypeAliasDecl {{.*}} alias_c1 'c1'
using alias_c1 = c1;
//CHECK: TypeAliasDecl {{.*}} alias_c1_ptr '__generic c1 *'
using alias_c1_ptr = c1 *;

struct c2 {
  alias_c1 y;
  alias_c1_ptr ptr = &y;
};


// Addr spaces for pointee of dependent types are not deduced
// during parsing but during template instantiation instead.

template <class T>
struct x1 {
//CHECK: -CXXMethodDecl {{.*}} operator= 'x1<T> &(const x1<T> &){{( __attribute__.*)?}} __generic'
//CHECK: -CXXMethodDecl {{.*}} operator= '__generic x1<int> &(const __generic x1<int> &){{( __attribute__.*)?}} __generic'
  x1<T>& operator=(const x1<T>& xx) {
    y = xx.y;
    return *this;
  }
  int y;
};

template <class T>
struct x2 {
//CHECK: -CXXMethodDecl {{.*}} foo 'void (x1<T> *){{( __attribute__.*)?}} __generic'
//CHECK: -CXXMethodDecl {{.*}} foo 'void (__generic x1<int> *){{( __attribute__.*)?}} __generic'
  void foo(x1<T>* xx) {
    m[0] = *xx;
  }
//CHECK: -FieldDecl {{.*}}  m 'x1<int> [2]'
  x1<T> m[2];
};

void bar(__global x1<int> *xx, __global x2<int> *bar) {
  bar->foo(xx);
}

template <typename T>
class x3 : public T {
public:
  //CHECK: -CXXConstructorDecl {{.*}} x3<T> 'void (const x3<T> &){{( __attribute__.*)?}} __generic'
  x3(const x3 &t);
};
//CHECK: -CXXConstructorDecl {{.*}} x3<T> 'void (const x3<T> &){{( __attribute__.*)?}} __generic'
template <typename T>
x3<T>::x3(const x3<T> &t) {}

template <class T>
T xxx(T *in) {
  // This pointer can't be deduced to generic because addr space
  // will be taken from the template argument.
  //CHECK: `-VarDecl {{.*}} i 'T *' cinit
  T *i = in;
  T ii;
  return *i;
}

__kernel void test() {
  int foo[10];
  xxx(&foo[0]);
}
