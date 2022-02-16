// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int code[]={0,0,0,0,1};
void foo(int x) {
  volatile int b;
  b = 0xffffffff;
}
void bar(int *pc) {
  static const void *l[] = {&&lab0, &&end};

  foo(0);
  goto *l[*pc];
 lab0:
  foo(0);
  pc++;
  goto *l[*pc];
 end:
  return;
}
int main(void) {
  bar(code);
  return 0;
}
