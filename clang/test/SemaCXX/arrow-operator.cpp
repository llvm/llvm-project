// RUN: %clang_cc1 -fsyntax-only -verify %s
struct T { 
  void f();
};

struct A {
  T* operator->(); // expected-note{{candidate function}}
};

struct B {
  T* operator->(); // expected-note{{candidate function}}
};

struct C : A, B {
};

struct D : A { };

struct E; // expected-note {{forward declaration of 'E'}}

void f(C &c, D& d, E& e) {
  c->f(); // expected-error{{use of overloaded operator '->' is ambiguous}}
  d->f();
  e->f(); // expected-error{{incomplete definition of type}}
}

// rdar://8875304
namespace rdar8875304 {
class Point {};
class Line_Segment{ public: Line_Segment(const Point&){} };
class Node { public: Point Location(){ Point p; return p; } };

void f()
{
   Node** node1;
   Line_Segment(node1->Location()); // expected-error {{not a structure or union}}
}
}


namespace arrow_suggest {

template <typename T>
class wrapped_ptr {
 public:
  wrapped_ptr(T* ptr) : ptr_(ptr) {}
  T* operator->() { return ptr_; }
  void Check(); // expected-note {{'Check' declared here}}
 private:
  T *ptr_;
};

class Worker {
 public:
  void DoSomething(); // expected-note {{'DoSomething' declared here}}
  void Chuck();
};

void test() {
  wrapped_ptr<Worker> worker(new Worker);
  worker.DoSomething(); // expected-error {{no member named 'DoSomething' in 'arrow_suggest::wrapped_ptr<arrow_suggest::Worker>'; did you mean to use '->' instead of '.'?}}
  worker.DoSamething(); // expected-error {{no member named 'DoSamething' in 'arrow_suggest::wrapped_ptr<arrow_suggest::Worker>'; did you mean to use '->' instead of '.'?}} \
                        // expected-error {{no member named 'DoSamething' in 'arrow_suggest::Worker'; did you mean 'DoSomething'?}}
  worker.Chuck(); // expected-error {{no member named 'Chuck' in 'arrow_suggest::wrapped_ptr<arrow_suggest::Worker>'; did you mean 'Check'?}}
}

} // namespace arrow_suggest

namespace no_crash_dependent_type {

template <class T>
struct A {
  void call();
  A *operator->();
};

template <class T>
void foo() {
  // The "requires an initializer" error seems unnecessary.
  A<int> &x = blah[7]; // expected-error {{use of undeclared identifier 'blah'}} \
                        // expected-error {{requires an initializer}}
  // x is dependent.
  x->call();
}

void test() {
  foo<int>(); // expected-note {{requested here}}
}

} // namespace no_crash_dependent_type

namespace clangd_issue_1073_no_crash_dependent_type {

template <typename T> struct Ptr {
  T *operator->();
};

struct Struct {
  int len;
};

template <int>
struct TemplateStruct {
  Ptr<Struct> val(); // expected-note {{declared here}}
};

template <int I>
void templateFunc(const TemplateStruct<I> &ts) {
  Ptr<Struct> ptr = ts.val(); // expected-error {{function is not marked const}}
  auto foo = ptr->len;
}

template void templateFunc<0>(const TemplateStruct<0> &); // expected-note {{requested here}}

} // namespace clangd_issue_1073_no_crash_dependent_type
