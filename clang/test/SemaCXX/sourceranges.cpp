// RUN: %clang_cc1 -triple i686-mingw32 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple i686-mingw32 -std=c++1z -ast-dump %s | FileCheck %s -check-prefix=CHECK-1Z

template<class T>
class P {
 public:
  P(T* t) {}
};

namespace foo {
class A { public: A(int = 0) {} };
enum B {};
typedef int C;
}

// CHECK: VarDecl {{0x[0-9a-fA-F]+}} <line:[[@LINE+1]]:1, col:36> col:15 ImplicitConstrArray 'foo::A [2]'
static foo::A ImplicitConstrArray[2];

int main() {
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::A *'
  P<foo::A> p14 = new foo::A;
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::B *'
  P<foo::B> p24 = new foo::B;
  // CHECK: CXXNewExpr {{0x[0-9a-fA-F]+}} <col:19, col:28> 'foo::C *'
  P<foo::C> pr4 = new foo::C;
}

foo::A getName() {
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:10, col:17> 'foo::A'
  return foo::A();
}

void destruct(foo::A *a1, foo::A *a2, P<int> *p1) {
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:8> '<bound member function type>' ->~A
  a1->~A();
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:16> '<bound member function type>' ->~A
  a2->foo::A::~A();
  // CHECK: MemberExpr {{0x[0-9a-fA-F]+}} <col:3, col:13> '<bound member function type>' ->~P
  p1->~P<int>();
}

struct D {
  D(int);
  ~D();
};

void construct() {
  using namespace foo;
  A a = A(12);
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:9, col:13> 'foo::A' 'void (int){{( __attribute__\(\(thiscall\)\))?}}'
  D d = D(12);
  // CHECK: CXXConstructExpr {{0x[0-9a-fA-F]+}} <col:9, col:13> 'D' 'void (int){{( __attribute__\(\(thiscall\)\))?}}'
}

void abort() __attribute__((noreturn));

namespace std {
typedef decltype(sizeof(int)) size_t;

template <typename E> struct initializer_list {
  const E *p;
  size_t n;
  initializer_list(const E *p, size_t n) : p(p), n(n) {}
};

template <typename F, typename S> struct pair {
  F f;
  S s;
  pair(const F &f, const S &s) : f(f), s(s) {}
};

struct string {
  const char *str;
  string() { abort(); }
  string(const char *S) : str(S) {}
  ~string() { abort(); }
};

template<typename K, typename V>
struct map {
  using T = pair<K, V>;
  map(initializer_list<T> i, const string &s = string()) {}
  ~map() { abort(); }
};

}; // namespace std

#if __cplusplus >= 201703L
// CHECK-1Z: FunctionDecl {{.*}} construct_with_init_list
std::map<int, int> construct_with_init_list() {
  // CHECK-1Z-NEXT: CompoundStmt
  // CHECK-1Z-NEXT: ReturnStmt {{.*}} <line:[[@LINE+5]]:3, col:35
  // CHECK-1Z-NEXT: ExprWithCleanups {{.*}} <col:10, col:35
  // CHECK-1Z-NEXT: CXXBindTemporaryExpr {{.*}} <col:10, col:35
  // CHECK-1Z-NEXT: CXXTemporaryObjectExpr {{.*}} <col:10, col:35
  // CHECK-1Z-NEXT: CXXStdInitializerListExpr {{.*}} <col:28, col:35
  return std::map<int, int>{{0, 0}};
}

// CHECK-1Z: NamespaceDecl {{.*}} in_class_init
namespace in_class_init {
  struct A {};

  // CHECK-1Z: CXXRecordDecl {{.*}} struct B definition
  struct B {
    // CHECK-1Z: FieldDecl {{.*}} a 'in_class_init::A'
    // CHECK-1Z-NEXT: InitListExpr {{.*}} <col:11, col:12
    A a = {};
  };
}

// CHECK-1Z: NamespaceDecl {{.*}} delegating_constructor_init
namespace delegating_constructor_init {
  struct A {};

  struct B : A {
    A a;
    B(A a) : a(a) {}
  };

  // CHECK-1Z: CXXRecordDecl {{.*}} struct C definition
  struct C : B {
    // CHECK-1Z: CXXConstructorDecl {{.*}} C
    // CHECK-1Z-NEXT: CXXCtorInitializer 'delegating_constructor_init::B'
    // CHECK-1Z-NEXT: CXXConstructExpr {{.*}} <col:11, col:15
    // CHECK-1Z-NEXT: InitListExpr {{.*}} <col:13, col:14
    C() : B({}) {};
  };
}

// CHECK-1Z: NamespaceDecl {{.*}} new_init
namespace new_init {
  void A() {
    // CHECK-1Z: CXXNewExpr {{.*}} <line:[[@LINE+2]]:5, col:14
    // CHECK-1Z-NEXT: InitListExpr {{.*}} <col:12, col:14
    new int{0};
  }
}
#endif
