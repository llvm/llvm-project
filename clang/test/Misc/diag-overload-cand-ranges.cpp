// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info %s 2>&1 | FileCheck %s --strict-whitespace
// CHECK:      error: no matching function
template <typename T> struct mcdata {
  typedef int result_type;
};
template <class T> typename mcdata<T>::result_type wrap_mean(mcdata<T> const &);
// CHECK:      :{[[@LINE+1]]:19-[[@LINE+1]]:53}: note: {{.*}}: no overload of 'wrap_mean'
void add_property(double (*)(mcdata<double> const &));
void f() { add_property(&wrap_mean); }

// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+1]]:10-[[@LINE+1]]:51}: note: {{.*}}: cannot pass pointer to generic address space
void baz(__attribute__((opencl_private)) int *Data) {}
void fizz() {
  int *Nop;
  baz(Nop);
  // CHECK:    error: no matching function
  // CHECK:    :[[@LINE+1]]:53: note: {{.*}}: 'this' object is in address space '__private'
  __attribute__((opencl_private)) static auto err = [&]() {};
  err();
}

// CHECK:      error: no matching function
struct Bar {
// CHECK:      :{[[@LINE+1]]:26-[[@LINE+1]]:32}: note: {{.*}} would lose const qualifier
static void foo(int num, int *X) {}
// CHECK:      :{[[@LINE+1]]:17-[[@LINE+1]]:25}: note: {{.*}} no known conversion
static void foo(int *err, int *x) {}
};
void bar(const int *Y) {
  Bar::foo(5, Y);
}

struct InComp;

struct A {};
struct B : public A{};
// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+5]]:36-[[@LINE+5]]:50}: note: {{.*}}: cannot convert initializer
// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+3]]:36-[[@LINE+3]]:50}: note: {{.*}}: cannot convert argument
// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+1]]:11-[[@LINE+1]]:18}: note: {{.*}}: no known conversion
void hoge(char aa, const char *bb, const A& third);

// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+1]]:14-[[@LINE+1]]:16}: note: {{.*}}: cannot convert from base class
void derived(B*);

void func(const A &arg) {
  hoge(1, "pass", {{{arg}}});
  InComp *a;
  hoge(1, "pass", a);
  hoge("first", 5, 6);
  A *b;
  derived(b);
}

struct Q {
  // CHECK:    error: invalid operands
  // CHECK:    :[[@LINE+1]]:6: note: {{.*}}: 'this' argument has type 'const Q'
  Q &operator+(void*);
};

void fuga(const Q q) { q + 3; }

template <short T> class Type1 {};
// CHECK:      error: no matching function
// CHECK:      :{[[@LINE+1]]:43-[[@LINE+1]]:54}: note: {{.*}}: expects an lvalue
template <short T> void Function1(int zz, Type1<T> &x, int ww) {}

void Function() { Function1(33, Type1<-42>(), 66); }
