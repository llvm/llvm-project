// RUN: %clang_cc1 -emit-llvm %s -o -

struct Evil {
 void fun ();
};
int foo();
typedef void (Evil::*memfunptr) ();
static memfunptr jumpTable[] = { &Evil::fun };

void Evil::fun() {
 (this->*jumpTable[foo()]) ();
}
