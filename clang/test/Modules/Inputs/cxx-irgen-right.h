#include "cxx-irgen-top.h"

inline int h() { return S<int>::f(); }

namespace ImplicitSpecialMembers {
  inline void create_right() {
    // Trigger declaration, but not definition, of special members.
    B b(0); C c(0); D d(0);
    // Trigger definition of move constructor.
    B b2(static_cast<B&&>(b));
    D d2(static_cast<D&&>(d));
  }
}
