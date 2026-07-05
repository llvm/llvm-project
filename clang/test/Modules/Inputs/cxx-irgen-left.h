#include "cxx-irgen-top.h"

S<int> s;

inline int instantiate_min() {
  return min(1, 2);
}

inline int instantiate_CtorInit(CtorInit<int> i = CtorInit<int>()) {
  return i.a;
}

namespace ImplicitSpecialMembers {
  inline void create_left() {
    // Trigger declaration, but not definition, of special members.
    B b(0); C c(0); D d(0);
    // Trigger definition of copy constructor.
    C c2(c); D d2(d);
  }
}

namespace OperatorDeleteLookup {
  // Trigger definition of A::~A() and lookup of operator delete.
  // Likewise for B<int>::~B().
  inline void f() { A a; B<int> b; }
}
