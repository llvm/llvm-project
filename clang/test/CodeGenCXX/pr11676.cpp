// RUN: %clang_cc1 %s -std=c++11 -emit-llvm-only
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
