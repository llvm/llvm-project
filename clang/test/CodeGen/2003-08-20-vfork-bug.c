// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

extern int vfork(void);
void test() {
  vfork();
}
