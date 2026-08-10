// RUN: %clang_cc1 %s -emit-llvm -o -
// REQUIRES: LP64

// Aggregates of size zero should be dropped from argument list.
typedef long int Tlong;
struct S2411 {
  __attribute__((aligned)) Tlong:0;
};

extern struct S2411 a2411[5];
extern void checkx2411(struct S2411);
void test2411(void) {
  checkx2411(a2411[0]);
}

// Proper handling of zero sized fields during type conversion.
typedef unsigned long long int Tal2ullong __attribute__((aligned(2)));
struct S2525 {
 Tal2ullong: 0;
 struct {
 } e;
};
struct S2525 s2525;

struct {
  signed char f;
  char :0;
  struct{}h;
  char * i[5];
} data; 

// Taking address of a zero sized field.
struct Z {};
struct Y {
  int i;
  struct Z z;
};
void *f(struct Y *y) {
  return &y->z;
}
