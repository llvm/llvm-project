// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -o /dev/null
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +egpr %s -o /dev/null

int foo(void) {
  register int a __asm__("ebx");
#ifdef __EGPR__
  register int b __asm__("r16");
#endif // __EGPR__
  return a;
}
