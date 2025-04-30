// BZ 12511
#include "tst-unique4.h"

template<int N>
int S<N>::i = N;
template<int N>
const int S<N>::j __attribute__ ((used)) = -1;

static int a[24] __attribute__ ((used)) =
  {
    S<1>::i, S<2>::i, S<3>::i, S<4>::i, S<5>::i, S<6>::i, S<7>::i, S<8>::i,
    S<9>::i, S<10>::i, S<11>::i, S<12>::i, S<13>::i, S<14>::i, S<15>::i,
    S<16>::i, S<17>::i, S<18>::i, S<19>::i, S<20>::i, S<21>::i, S<22>::i,
    S<23>::i, S<24>::i
  };

static int b __attribute__ ((used)) = S<1>::j;
