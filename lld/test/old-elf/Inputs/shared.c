#include <stdio.h>

extern int i;
int i = 42;

// Undefined weak function in a dynamic library.
__attribute__((weak)) void weakfoo();

// Regular function in a dynamic library.
void foo() {
  // Try to call weakfoo so that the reference to weekfoo will be included in
  // the resulting .so file.
  if (weakfoo)
    weakfoo();
  puts("Fooo!!");
}
