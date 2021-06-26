// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s

typedef __typeof__(sizeof(0)) size_t;

// Declare an 'operator new' template to tickle a bug in __builtin_operator_new.
template<typename T> void *operator new(size_t, int (*)(T));

// Ensure that this declaration doesn't cause operator new to lose its
// 'noalias' attribute.
void *operator new[](size_t);

void t1() {
  delete new int;
  delete [] new int [3];
}

// CHECK: declare nonnull i8* @_Znwm(i64) [[ATTR_NOBUILTIN:#[^ ]*]]
// CHECK: declare void @_ZdlPv(i8*) [[ATTR_NOBUILTIN_NOUNWIND:#[^ ]*]]
// CHECK: declare nonnull i8* @_Znam(i64) [[ATTR_NOBUILTIN]]
// CHECK: declare void @_ZdaPv(i8*) [[ATTR_NOBUILTIN_NOUNWIND]]

namespace std {
  struct nothrow_t {};
}
std::nothrow_t nothrow;

// Declare the reserved placement operators.
void *operator new(size_t, void*) throw();
void operator delete(void*, void*) throw();
void *operator new[](size_t, void*) throw();
void operator delete[](void*, void*) throw();

// Declare the replaceable global allocation operators.
void *operator new(size_t, const std::nothrow_t &) throw();
void *operator new[](size_t, const std::nothrow_t &) throw();
void operator delete(void *, const std::nothrow_t &) throw();
void operator delete[](void *, const std::nothrow_t &) throw();

// Declare some other placemenet operators.
void *operator new(size_t, void*, bool) throw();
void *operator new[](size_t, void*, bool) throw();

void t2(int* a) {
  int* b = new (a) int;
}

struct S {
  int a;
};

// POD types.
void t3() {
  int *a = new int(10);
  _Complex int* b = new _Complex int(10i);

  S s;
  s.a = 10;
  S *sp = new S(s);
}

// Non-POD
struct T {
  T();
  int a;
};

void t4() {
  // CHECK: call void @_ZN1TC1Ev
  T *t = new T;
}

struct T2 {
  int a;
  T2(int, int);
};

void t5() {
  // CHECK: call void @_ZN2T2C1Eii
  T2 *t2 = new T2(10, 10);
}

int *t6() {
  // Null check.
  return new (0) int(10);
}

void t7() {
  new int();
}

struct U {
  ~U();
};

void t8(int n) {
  new int[10];
  new int[n];

  // Non-POD
  new T[10];
  new T[n];

  // Cookie required
  new U[10];
  new U[n];
}

void t9() {
  bool b;

  new bool(true);
  new (&b) bool(true);
}

struct A {
  void* operator new(__typeof(sizeof(int)), int, float, ...);
  A();
};

A* t10() {
   // CHECK: @_ZN1AnwEmifz
  return new(1, 2, 3.45, 100) A;
}

// CHECK-LABEL: define{{.*}} void @_Z3t11i
struct B { int a; };
struct Bmemptr { int Bmemptr::* memptr; int a; };

void t11(int n) {
  // CHECK: call noalias nonnull i8* @_Znwm
  // CHECK: call void @llvm.memset.p0i8.i64(
  B* b = new B();

  // CHECK: call noalias nonnull i8* @_Znam
  // CHECK: {{call void.*llvm.memset.p0i8.i64.*i8 0, i64 %}}
  B *b2 = new B[n]();

  // CHECK: call noalias nonnull i8* @_Znam
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  // CHECK: br
  Bmemptr *b_memptr = new Bmemptr[n]();

  // CHECK: ret void
}

struct Empty { };

// We don't need to initialize an empty class.
// CHECK-LABEL: define{{.*}} void @_Z3t12v
void t12() {
  // CHECK: call noalias nonnull i8* @_Znam
  // CHECK-NOT: br
  (void)new Empty[10];

  // CHECK: call noalias nonnull i8* @_Znam
  // CHECK-NOT: br
  (void)new Empty[10]();

  // CHECK: ret void
}

// Zero-initialization
// CHECK-LABEL: define{{.*}} void @_Z3t13i
void t13(int n) {
  // CHECK: call noalias nonnull i8* @_Znwm
  // CHECK: store i32 0, i32*
  (void)new int();

  // CHECK: call noalias nonnull i8* @_Znam
  // CHECK: {{call void.*llvm.memset.p0i8.i64.*i8 0, i64 %}}
  (void)new int[n]();

  // CHECK-NEXT: ret void
}

struct Alloc{
  int x;
  void* operator new[](size_t size);
  __attribute__((returns_nonnull)) void *operator new[](size_t size, const std::nothrow_t &) throw();
  void operator delete[](void* p);
  ~Alloc();
};

void f() {
  // CHECK: call i8* @_ZN5AllocnaEm(i64 808)
  // CHECK: store i64 200
  // CHECK: call void @_ZN5AllocD1Ev(
  // CHECK: call void @_ZN5AllocdaEPv(i8*
  delete[] new Alloc[10][20];
  // CHECK: [[P:%.*]] = call nonnull i8* @_ZN5AllocnaEmRKSt9nothrow_t(i64 808, {{.*}}) [[ATTR_NOUNWIND:#[^ ]*]]
  // CHECK-NOT: icmp eq i8* [[P]], null
  // CHECK: store i64 200
  delete[] new (nothrow) Alloc[10][20];
  // CHECK: call noalias nonnull i8* @_Znwm
  // CHECK: call void @_ZdlPv(i8*
  delete new bool;
  // CHECK: ret void
}

namespace test15 {
  struct A { A(); ~A(); };

  // CHECK-LABEL:    define{{.*}} void @_ZN6test156test0aEPv(
  // CHECK:      [[P:%.*]] = load i8*, i8**
  // CHECK-NOT:  icmp eq i8* [[P]], null
  // CHECK-NOT:  br i1
  // CHECK:      [[T0:%.*]] = bitcast i8* [[P]] to [[A:%.*]]*
  // CHECK-NEXT: call void @_ZN6test151AC1Ev([[A]]* {{[^,]*}} [[T0]])
  void test0a(void *p) {
    new (p) A();
  }

  // CHECK-LABEL:    define{{.*}} void @_ZN6test156test0bEPv(
  // CHECK:      [[P0:%.*]] = load i8*, i8**
  // CHECK:      [[P:%.*]] = call i8* @_ZnwmPvb(i64 1, i8* [[P0]]
  // CHECK-NEXT: icmp eq i8* [[P]], null
  // CHECK-NEXT: br i1
  // CHECK:      [[T0:%.*]] = bitcast i8* [[P]] to [[A:%.*]]*
  // CHECK-NEXT: call void @_ZN6test151AC1Ev([[A]]* {{[^,]*}} [[T0]])
  void test0b(void *p) {
    new (p, true) A();
  }

  // CHECK-LABEL:    define{{.*}} void @_ZN6test156test1aEPv(
  // CHECK:      [[P:%.*]] = load i8*, i8**
  // CHECK-NOT:  icmp eq i8* [[P]], null
  // CHECK-NOT:  br i1
  // CHECK:      [[BEGIN:%.*]] = bitcast i8* [[P]] to [[A:%.*]]*
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[BEGIN]], i64 5
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: call void @_ZN6test151AC1Ev([[A]]* {{[^,]*}} [[CUR]])
  // CHECK-NEXT: [[NEXT]] = getelementptr inbounds [[A]], [[A]]* [[CUR]], i64 1
  // CHECK-NEXT: [[DONE:%.*]] = icmp eq [[A]]* [[NEXT]], [[END]]
  // CHECK-NEXT: br i1 [[DONE]]
  void test1a(void *p) {
    new (p) A[5];
  }

  // CHECK-LABEL:    define{{.*}} void @_ZN6test156test1bEPv(
  // CHECK:      [[P0:%.*]] = load i8*, i8**
  // CHECK:      [[P:%.*]] = call i8* @_ZnamPvb(i64 13, i8* [[P0]]
  // CHECK-NEXT: icmp eq i8* [[P]], null
  // CHECK-NEXT: br i1
  // CHECK:      [[AFTER_COOKIE:%.*]] = getelementptr inbounds i8, i8* [[P]], i64 8
  // CHECK:      [[BEGIN:%.*]] = bitcast i8* [[AFTER_COOKIE]] to [[A:%.*]]*
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[BEGIN]], i64 5
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: call void @_ZN6test151AC1Ev([[A]]* {{[^,]*}} [[CUR]])
  // CHECK-NEXT: [[NEXT]] = getelementptr inbounds [[A]], [[A]]* [[CUR]], i64 1
  // CHECK-NEXT: [[DONE:%.*]] = icmp eq [[A]]* [[NEXT]], [[END]]
  // CHECK-NEXT: br i1 [[DONE]]
  void test1b(void *p) {
    new (p, true) A[5];
  }

  // TODO: it's okay if all these size calculations get dropped.
  // FIXME: maybe we should try to throw on overflow?
  // CHECK-LABEL:    define{{.*}} void @_ZN6test155test2EPvi(
  // CHECK:      [[N:%.*]] = load i32, i32*
  // CHECK-NEXT: [[T0:%.*]] = sext i32 [[N]] to i64
  // CHECK-NEXT: [[P:%.*]] = load i8*, i8**
  // CHECK:      [[BEGIN:%.*]] = bitcast i8* [[P]] to [[A:%.*]]*
  // CHECK-NEXT: [[ISEMPTY:%.*]] = icmp eq i64 [[T0]], 0
  // CHECK-NEXT: br i1 [[ISEMPTY]],
  // CHECK:      [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[BEGIN]], i64 [[T0]]
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]],
  // CHECK-NEXT: call void @_ZN6test151AC1Ev([[A]]* {{[^,]*}} [[CUR]])
  void test2(void *p, int n) {
    new (p) A[n];
  }
}

namespace PR10197 {
  // CHECK-LABEL: define weak_odr void @_ZN7PR101971fIiEEvv()
  template<typename T>
  void f() {
    // CHECK: [[CALL:%.*]] = call noalias nonnull i8* @_Znwm
    // CHECK-NEXT: [[CASTED:%.*]] = bitcast i8* [[CALL]] to
    new T;
    // CHECK-NEXT: ret void
  }

  template void f<int>();
}

namespace PR11523 {
  class MyClass;
  typedef int MyClass::* NewTy;
  // CHECK-LABEL: define{{.*}} i64* @_ZN7PR115231fEv
  // CHECK: store i64 -1
  NewTy* f() { return new NewTy[2](); }
}

namespace PR11757 {
  // Make sure we elide the copy construction.
  struct X { X(); X(const X&); };
  X* a(X* x) { return new X(X()); }
  // CHECK: define {{.*}} @_ZN7PR117571aEPNS_1XE
  // CHECK: [[CALL:%.*]] = call noalias nonnull i8* @_Znwm
  // CHECK-NEXT: [[CASTED:%.*]] = bitcast i8* [[CALL]] to
  // CHECK-NEXT: call void @_ZN7PR117571XC1Ev({{.*}}* {{[^,]*}} [[CASTED]])
  // CHECK-NEXT: ret {{.*}} [[CASTED]]
}

namespace PR13380 {
  struct A { A() {} };
  struct B : public A { int x; };
  // CHECK-LABEL: define{{.*}} i8* @_ZN7PR133801fEv
  // CHECK: call noalias nonnull i8* @_Znam(
  // CHECK: call void @llvm.memset.p0i8
  // CHECK-NEXT: call void @_ZN7PR133801BC1Ev
  void* f() { return new B[2](); }
}

struct MyPlacementType {} mpt;
void *operator new(size_t, MyPlacementType);

namespace N3664 {
  struct S { S() throw(int); };

  // CHECK-LABEL: define{{.*}} void @_ZN5N36641fEv
  void f() {
    // CHECK: call noalias nonnull i8* @_Znwm(i64 4) [[ATTR_BUILTIN_NEW:#[^ ]*]]
    int *p = new int; // expected-note {{allocated with 'new' here}}
    // CHECK: call void @_ZdlPv({{.*}}) [[ATTR_BUILTIN_DELETE:#[^ ]*]]
    delete p;

    // CHECK: call noalias nonnull i8* @_Znam(i64 12) [[ATTR_BUILTIN_NEW]]
    int *q = new int[3];
    // CHECK: call void @_ZdaPv({{.*}}) [[ATTR_BUILTIN_DELETE]]
    delete[] p; // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}

    // CHECK: call noalias i8* @_ZnamRKSt9nothrow_t(i64 3, {{.*}}) [[ATTR_NOBUILTIN_NOUNWIND_ALLOCSIZE:#[^ ]*]]
    (void) new (nothrow) S[3];

    // CHECK: call i8* @_Znwm15MyPlacementType(i64 4){{$}}
    (void) new (mpt) int;
  }

  // CHECK: declare i8* @_ZnamRKSt9nothrow_t(i64, {{.*}}) [[ATTR_NOBUILTIN_NOUNWIND_ALLOCSIZE:#[^ ]*]]

  // CHECK-LABEL: define{{.*}} void @_ZN5N36641gEv
  void g() {
    // It's OK for there to be attributes here, so long as we don't have a
    // 'builtin' attribute.
    // CHECK: call noalias nonnull i8* @_Znwm(i64 4) {{#[^ ]*}}{{$}}
    int *p = (int*)operator new(4);
    // CHECK: call void @_ZdlPv({{.*}}) [[ATTR_NOUNWIND:#[^ ]*]]
    operator delete(p);

    // CHECK: call noalias nonnull i8* @_Znam(i64 12) {{#[^ ]*}}{{$}}
    int *q = (int*)operator new[](12);
    // CHECK: call void @_ZdaPv({{.*}}) [[ATTR_NOUNWIND]]
    operator delete [](p);

    // CHECK: call noalias i8* @_ZnamRKSt9nothrow_t(i64 3, {{.*}}) [[ATTR_NOUNWIND_ALLOCSIZE:#[^ ]*]]
    (void) operator new[](3, nothrow);
  }
}

namespace builtins {
  // CHECK-LABEL: define{{.*}} void @_ZN8builtins1fEv
  void f() {
    // CHECK: call noalias nonnull i8* @_Znwm(i64 4) [[ATTR_BUILTIN_NEW]]
    // CHECK: call void @_ZdlPv({{.*}}) [[ATTR_BUILTIN_DELETE]]
    __builtin_operator_delete(__builtin_operator_new(4));
  }
}

// CHECK-DAG: attributes [[ATTR_NOBUILTIN]] = {{[{].*}} nobuiltin allocsize(0) {{.*[}]}}
// CHECK-DAG: attributes [[ATTR_NOBUILTIN_NOUNWIND]] = {{[{].*}} nobuiltin nounwind {{.*[}]}}
// CHECK-DAG: attributes [[ATTR_NOBUILTIN_NOUNWIND_ALLOCSIZE]] = {{[{].*}} nobuiltin nounwind allocsize(0) {{.*[}]}}

// CHECK-DAG: attributes [[ATTR_BUILTIN_NEW]] = {{[{].*}} builtin {{.*[}]}}
// CHECK-DAG: attributes [[ATTR_BUILTIN_DELETE]] = {{[{].*}} builtin {{.*[}]}}

// The ([^b}|...) monstrosity is matching a character that's not the start of 'builtin'.
// Add more letters if this matches some other attribute.
// CHECK-DAG: attributes [[ATTR_NOUNWIND]] = {{([^b]|b[^u]|bu[^i]|bui[^l])*}} nounwind {{([^b]|b[^u]|bu[^i]|bui[^l])*$}}
// CHECK-DAG: attributes [[ATTR_NOUNWIND_ALLOCSIZE]] = {{([^b]|b[^u]|bu[^i]|bui[^l])*}} nounwind allocsize(0) {{([^b]|b[^u]|bu[^i]|bui[^l])*$}}
