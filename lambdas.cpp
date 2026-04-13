namespace {
int blah = 5;
}

namespace {
template <typename T> void tfunc(const T &) {}
inline void x() {
  tfunc([] {});      // GCC: tfunc<x()::<lambda()> >
                     // Dem: tfunc<x()::'lambda'()>
  tfunc([] {});      // GCC: tfunc<x()::<lambda()> >
                     // Dem: tfunc<x()::'lambda0'()>
  tfunc([](int) {}); // GCC: tfunc<x()::<lambda(int)> >
                     // Dem: tfunc<x()::'lambda'(int)>
}
} // namespace

inline int i = (tfunc([] {}),      // GCC: tfunc<<lambda()> >
                                   // Dem: tfunc<i::'lambda'()>
                tfunc([] {}),      // GCC: tfunc<<lambda()> >
                                   // Dem: tfunc<i::'lambda0'()>
                tfunc([](int) {}), // GCC: tfunc<<lambda(int)> >
                                   // Dem:  tfunc<i::'lambda'(int)>
                1);
void y() {
  x();
  (void)i;
  tfunc([] {});      // GCC: tfunc<y()::<lambda()> >
                     // Dem: tfunc<y()::'lambda'()>
  tfunc([] {});      // GCC: tfunc<y()::<lambda()> >
                     // Dem: tfunc<y()::'lambda0'()>
  tfunc([](int) {}); // GCC: tfunc<y()::<lambda(int)> >
                     // Dem: tfunc<y()::'lambda'(int)>
  tfunc([](int) {}); // GCC: tfunc<y()::<lambda(int)> >
                     // Dem: tfunc<y()::'lambda'(int)>
}
