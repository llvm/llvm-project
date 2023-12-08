#include "c.h"

#ifndef OTHER_H
#define OTHER_H
namespace N {
  struct C {
    template<typename U> static void f(U) {}
  };
}
#endif
