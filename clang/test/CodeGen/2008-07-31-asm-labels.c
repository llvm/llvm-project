// RUN: %clang_cc1 -emit-llvm -o - %s |FileCheck %s -check-prefixes CHECK,CHECKREF
// RUN: %clang_cc1 -DUSE_DEF -emit-llvm -o - %s |FileCheck %s -check-prefixes CHECK,CHECKDEF
// <rdr://6116729>

//CHECK: _renamed{{.*}} = external {{.*}}global
void pipe() asm("_thisIsNotAPipe");

void f0(void) {
  pipe();
//CHECK: call {{.*}}_thisIsNotAPipe
}

void pipe(int);
//CHECKREF: declare {{.*}}_thisIsNotAPipe

void f1(void) {
  pipe(1);
//CHECK: call {{.*}}_thisIsNotAPipe
}

#ifdef USE_DEF
//CHECKDEF: define {{.*}}_thisIsNotAPipe
void pipe(int arg) {
  int x = 10;
}
#endif

// PR3698
extern int g0 asm("_renamed");
int f2(void) {
  return g0;
//CHECK: load {{.*}}_renamed
}
