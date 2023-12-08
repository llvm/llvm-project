// RUN: %clang_cc1 -emit-llvm %s -triple x86_64-apple-darwin -o /dev/null


struct foo { int X; };
struct bar { int Y; };

extern int Func(struct foo*) __asm__("Func64");
extern int Func64(struct bar*);

int Func(struct foo *F) {
  return 1;
}

int Func64(struct bar* B) {
  return 0;
}


int test(void) {
  Func(0);    /* should be renamed to call Func64 */
  Func64(0);
}
