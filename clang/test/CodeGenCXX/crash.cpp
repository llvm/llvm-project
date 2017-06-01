// RUN: %clang_cc1 %s -std=c++11 -emit-llvm-only
// RUN: %clang_cc1 -emit-obj -o %t -debug-info-kind=line-tables-only -std=c++11 %s
// CHECK that we don't crash.

// PR11676's example is ill-formed:
/*
union _XEvent {
};
void ProcessEvent() {
  _XEvent pluginEvent = _XEvent();
}
*/

// Example from PR11665:
void f() {
  union U { int field; } u = U();
  (void)U().field;
}

namespace PR17476 {
struct string {
  string(const char *__s);
  string &operator+=(const string &__str);
};

template <class ELFT> void finalizeDefaultAtomValues() {
  auto startEnd = [&](const char * sym)->void {
    string start("__");
    start += sym;
  }
  ;
  startEnd("preinit_array");
}

void f() { finalizeDefaultAtomValues<int>(); }
}
