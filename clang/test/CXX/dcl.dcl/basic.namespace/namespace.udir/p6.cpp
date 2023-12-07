// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef int pid_t;
namespace ns {
  typedef int pid_t;
}
using namespace ns;
pid_t x;

struct A { };
namespace ns {
  typedef ::A A;
}
A a;
