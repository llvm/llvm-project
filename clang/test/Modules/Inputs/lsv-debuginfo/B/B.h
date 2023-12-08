#ifndef B_H
#define B_H
#include <A/ADT.h>
#include <C/C.h>

namespace llvm {
struct S {
  unsigned a, b, c, d;
};
class C {
  Optional<S> S;
};
}
#endif
