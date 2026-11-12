// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

// expected-error@+1{{invalid option 'invalid' for cpu_dispatch}}
void __attribute__((cpu_dispatch(atom, invalid))) invalid_cpu();

void __attribute__((cpu_specific(atom))) no_default(void);
void __attribute__((cpu_specific(sandybridge)))  no_default(void);

struct MVReference {
  int __attribute__((cpu_specific(sandybridge))) bar(void);
  int __attribute__((cpu_specific(ivybridge))) bar(void);
  int __attribute__((cpu_specific(sandybridge))) foo(void);
};

void use1(void){
  // OK, will fail in the linker, unless another TU provides the cpu_dispatch.
  no_default();

  // expected-error@+1 {{call to non-static member function without an object argument}}
  +MVReference::bar;
  // expected-error@+1 {{call to non-static member function without an object argument}}
  +MVReference::foo;
  // expected-error@+1 {{reference to multiversioned function could not be resolved; did you mean to call it?}}
  &MVReference::bar;
  // expected-error@+1 {{reference to multiversioned function could not be resolved; did you mean to call it?}}
  &MVReference::foo;
}

//expected-error@+1 {{attribute 'cpu_specific' multiversioned functions do not yet support constexpr functions}}
constexpr int __attribute__((cpu_specific(sandybridge))) foo(void);

int __attribute__((cpu_specific(sandybridge))) foo2(void);
//expected-error@+1 {{attribute 'cpu_specific' multiversioned functions do not yet support constexpr functions}}
constexpr int __attribute__((cpu_specific(ivybridge))) foo2(void);

static int __attribute__((cpu_specific(sandybridge))) bar(void);
//expected-error@+1 {{multiversioned function declaration has a different linkage}}
int __attribute__((cpu_dispatch(ivybridge))) bar(void) {}

// OK
extern int __attribute__((cpu_specific(sandybridge))) bar2(void);
int __attribute__((cpu_dispatch(ivybridge))) bar2(void) {}

namespace {
int __attribute__((cpu_specific(sandybridge))) bar3(void);
static int __attribute__((cpu_dispatch(ivybridge))) bar3(void) {}
}


inline int __attribute__((cpu_specific(sandybridge))) baz(void);
//expected-error@+1 {{multiversioned function declaration has a different inline specification}}
int __attribute__((cpu_specific(ivybridge))) baz(void) {return 1;}

void __attribute__((cpu_specific(atom))) diff_return(void);
//expected-error@+1 {{multiversioned function declaration has a different return type}}
int __attribute__((cpu_specific(sandybridge))) diff_return(void);

int __attribute__((cpu_specific(atom))) diff_noexcept(void) noexcept(true);
//expected-error@+2 {{exception specification in declaration does not match previous declaration}}
//expected-note@-2 {{previous declaration is here}}
int __attribute__((cpu_specific(sandybridge))) diff_noexcept(void) noexcept(false);

// FIXME: Add support for templates and virtual functions!
// expected-error@+2 {{multiversioned functions do not yet support function templates}}
template<typename T>
int __attribute__((cpu_specific(atom))) foo(T) { return 0; }
// expected-error@+2 {{multiversioned functions do not yet support function templates}}
template<typename T>
int __attribute__((cpu_specific(sandybridge))) foo2(T);

struct S {
  // expected-error@+2 {{multiversioned functions do not yet support function templates}}
  template<typename T>
  int __attribute__((cpu_specific(atom))) foo(T) { return 0; }

  // expected-error@+2 {{multiversioned functions do not yet support function templates}}
  template<typename T>
  int __attribute__((cpu_dispatch(ivybridge))) foo2(T) {}

  // expected-error@+1 {{multiversioned functions do not yet support virtual functions}}
  virtual void __attribute__((cpu_specific(atom))) virt();
};

extern "C" {
int __attribute__((cpu_specific(atom))) diff_mangle(void) { return 0; }
}
//expected-error@+1 {{multiversioned function declaration has a different language linkage}}
int __attribute__((cpu_specific(sandybridge))) diff_mangle(void) { return 0; }

__attribute__((cpu_specific(atom))) void DiffDecl();
namespace N {
using ::DiffDecl;
// expected-error@+3 {{declaration conflicts with target of using declaration already in scope}}
// expected-note@-4 {{target of using declaration}}
// expected-note@-3 {{using declaration}}
__attribute__((cpu_dispatch(atom))) void DiffDecl();
} // namespace N

struct SpecialFuncs {
  // expected-error@+1 {{multiversioned functions do not yet support constructors}}
  __attribute__((cpu_specific(atom))) SpecialFuncs();
  // expected-error@+1 {{multiversioned functions do not yet support destructors}}
  __attribute__((cpu_specific(atom))) ~SpecialFuncs();

  // expected-error@+1 {{multiversioned functions do not yet support defaulted functions}}
  SpecialFuncs& __attribute__((cpu_specific(atom))) operator=(const SpecialFuncs&) = default;
  // expected-error@+1 {{multiversioned functions do not yet support deleted functions}}
  SpecialFuncs& __attribute__((cpu_specific(atom))) operator=(SpecialFuncs&&) = delete;
};

struct OutOfLine {
  int __attribute__((cpu_specific(atom, ivybridge))) foo(int);
};

int __attribute__((cpu_specific(atom, ivybridge))) OutOfLine::foo(int) { return 0; }
int __attribute__((cpu_specific(sandybridge))) OutOfLine::foo(int) { return 1; }

// Ensure Cpp Spelling works.
[[clang::cpu_specific(ivybridge,atom)]] int CppSpelling(){}

// expected-error@+1 {{lambda cannot be declared 'cpu_dispatch'}}
auto x = []() __attribute__((cpu_dispatch(atom))) {};
