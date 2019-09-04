#include "CppLib.h"

void g() {
  throw 0;
}

void f() {
  try {
    g();
  } catch (...) {
  }
}
