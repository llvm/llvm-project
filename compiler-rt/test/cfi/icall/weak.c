// Test that weak symbols stay weak.
// RUN: %clang_cfi -lm -o %t1 %s && %t1
// XFAIL: darwin

__attribute__((weak)) void does_not_exist(void);

__attribute__((noinline))
void foo(void (*p)(void)) {
  p();
}

int main(int argc, char **argv) {
  if (does_not_exist)
    foo(does_not_exist);
}
