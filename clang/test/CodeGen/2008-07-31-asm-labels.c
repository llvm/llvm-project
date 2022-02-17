// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: not grep "@pipe()" %t
// RUN: grep '_thisIsNotAPipe' %t | count 3
// RUN: not grep '@g0' %t
// RUN: grep '_renamed' %t | count 2
// RUN: %clang_cc1 -DUSE_DEF -emit-llvm -o %t %s
// RUN: not grep "@pipe()" %t
// RUN: grep '_thisIsNotAPipe' %t | count 3
// <rdr://6116729>

void pipe() asm("_thisIsNotAPipe");

void f0(void) {
  pipe();
}

void pipe(int);

void f1(void) {
  pipe(1);
}

#ifdef USE_DEF
void pipe(int arg) {
  int x = 10;
}
#endif

// PR3698
extern int g0 asm("_renamed");
int f2(void) {
  return g0;
}
