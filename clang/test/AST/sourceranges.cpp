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

// CHECK: VarDecl {{0x[0-9a-fA-F]+}} <line:[[@LINE+1]]:1, col:36> col:15 ImplicitConstrArray 'foo::A[2]'
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

namespace PR38987 {
struct A { A(); };
template <class T> void f() { T{}; }
template void f<A>();
// CHECK: CXXTemporaryObjectExpr {{.*}} <col:31, col:33> 'PR38987::A':'PR38987::A'
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

// CHECK: NamespaceDecl {{.*}} attributed_decl
namespace attributed_decl {
  void f() {
    // CHECK: DeclStmt {{.*}} <line:[[@LINE+1]]:5, col:28>
    [[maybe_unused]] int i1;
    // CHECK: DeclStmt {{.*}} <line:[[@LINE+1]]:5, col:35>
    __attribute__((unused)) int i2;
    // CHECK: DeclStmt {{.*}} <line:[[@LINE+1]]:5, col:35>
    int __attribute__((unused)) i3;
    // CHECK: DeclStmt {{.*}} <<built-in>:{{.*}}, {{.*}}:[[@LINE+1]]:40>
    __declspec(dllexport) extern int i4;
    // CHECK: DeclStmt {{.*}} <line:[[@LINE+1]]:5, col:40>
    extern int __declspec(dllexport) i5;
  }
}

// CHECK-1Z: NamespaceDecl {{.*}} attributed_case
namespace attributed_case {
void f(int n) {
  switch (n) {
  case 0:
    n--;
    // CHECK: AttributedStmt {{.*}} <line:[[@LINE+2]]:5, line:[[@LINE+4]]:35>
    // CHECK: FallThroughAttr {{.*}} <line:[[@LINE+1]]:20>
    __attribute__((fallthrough))
    // CHECK: FallThroughAttr {{.*}} <line:[[@LINE+1]]:22>
      __attribute__((fallthrough));
  case 1:
    n++;
    break;
  }
}
} // namespace attributed_case

// CHECK: NamespaceDecl {{.*}} attributed_stmt
namespace attributed_stmt {
  // In DO_PRAGMA and _Pragma cases, `LoopHintAttr` comes from <scratch space>
  // file.

  #define DO_PRAGMA(x) _Pragma (#x)

  void f() {
    // CHECK: AttributedStmt {{.*}} <line:[[@LINE-3]]:24, line:[[@LINE+2]]:33>
    DO_PRAGMA (unroll(2))
    for (int i = 0; i < 10; ++i);

    // CHECK: AttributedStmt {{.*}} <line:[[@LINE+2]]:5, line:[[@LINE+3]]:33>
    // CHECK: LoopHintAttr {{.*}} <line:[[@LINE+1]]:13, col:22>
    #pragma unroll(2)
    for (int i = 0; i < 10; ++i);

    // CHECK: AttributedStmt {{.*}} <line:[[@LINE+2]]:5, line:[[@LINE+5]]:33>
    // CHECK: LoopHintAttr {{.*}} <line:[[@LINE+1]]:19, col:41>
    #pragma clang loop vectorize(enable)
    // CHECK: LoopHintAttr {{.*}} <line:[[@LINE+1]]:19, col:42>
    #pragma clang loop interleave(enable)
    for (int i = 0; i < 10; ++i);

    // CHECK: AttributedStmt {{.*}} <line:[[@LINE+1]]:5, line:[[@LINE+2]]:33>
    _Pragma("unroll(2)")
    for (int i = 0; i < 10; ++i);
  }
}

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
