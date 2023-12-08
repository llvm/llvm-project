// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -fms-extensions -verify %s -std=c++11

// The MS ABI has a few ways to generate constructor closures, which require
// instantiating and checking the semantics of default arguments. Make sure we
// do that right.

template <typename T>
struct DependentDefaultCtorArg {
  // expected-error@+1 {{type 'int' cannot be used prior to '::' because it has no members}}
  DependentDefaultCtorArg(int n = T::error);
};
struct
__declspec(dllexport) // expected-note {{due to 'ExportDefaultCtorClosure' being dllexported}}
ExportDefaultCtorClosure // expected-note {{in instantiation of default function argument expression for 'DependentDefaultCtorArg<int>' required here}} expected-note {{implicit default constructor for 'ExportDefaultCtorClosure' first required here}}
: DependentDefaultCtorArg<int>
{};

template <typename T>
struct DependentDefaultCopyArg {
  DependentDefaultCopyArg() {}
  // expected-error@+1 {{type 'int' cannot be used prior to '::' because it has no members}}
  DependentDefaultCopyArg(const DependentDefaultCopyArg &o, int n = T::member) {}
};

struct HasMember {
  enum { member = 0 };
};
void UseDependentArg() { throw DependentDefaultCopyArg<HasMember>(); }

void ErrorInDependentArg() {
  throw DependentDefaultCopyArg<int>(); // expected-note {{required here}}
}

struct HasCleanup {
  ~HasCleanup();
};

struct Default {
  Default(const Default &o, int d = (HasCleanup(), 42));
};

void f(const Default &d) {
  throw d;
}
