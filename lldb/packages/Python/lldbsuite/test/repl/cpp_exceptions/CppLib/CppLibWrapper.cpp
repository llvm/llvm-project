#include "CppLib.h"
extern "C" {
void f_with_exceptions(void);
void f_with_exceptions(void) {
  f();
}
}
