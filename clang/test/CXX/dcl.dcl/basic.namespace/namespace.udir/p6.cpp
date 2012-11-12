// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// <rdar://problem/8296180>
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
