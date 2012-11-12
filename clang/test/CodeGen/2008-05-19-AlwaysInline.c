// RUN: %clang_cc1 %s -emit-llvm -fno-unit-at-a-time -O0 -o - | not grep sabrina
// RUN: %clang_cc1 %s -emit-llvm -funit-at-a-time -O0 -o - | not grep sabrina

static inline int sabrina (void) __attribute__((always_inline));
static inline int sabrina (void)
{
  return 13;
}
int bar (void)
{
  return sabrina () + 68;
}
