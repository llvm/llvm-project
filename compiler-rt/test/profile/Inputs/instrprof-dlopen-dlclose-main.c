#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  dlerror();
  void *f1_handle = dlopen("func.shared", RTLD_LAZY | RTLD_GLOBAL);
  if (f1_handle == NULL) {
    fprintf(stderr, "unable to open 'func.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  void (*func)(void) = (void (*)(void))dlsym(f1_handle, "func");
  if (func == NULL) {
    fprintf(stderr, "unable to lookup symbol 'func': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  dlerror();
  void *f2_handle = dlopen("func2.shared", RTLD_LAZY | RTLD_GLOBAL);
  if (f2_handle == NULL) {
    fprintf(stderr, "unable to open 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  void (*func2)(void) = (void (*)(void))dlsym(f2_handle, "func2");
  if (func2 == NULL) {
    fprintf(stderr, "unable to lookup symbol 'func2': %s\n", dlerror());
    return EXIT_FAILURE;
  }
  func2();

#ifdef USE_LIB3
  void *f3_handle = dlopen("func3.shared", RTLD_LAZY | RTLD_GLOBAL);
  if (f3_handle == NULL) {
    fprintf(stderr, "unable to open 'func3.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  void (*func3)(void) = (void (*)(void))dlsym(f3_handle, "func3");
  if (func3 == NULL) {
    fprintf(stderr, "unable to lookup symbol 'func3': %s\n", dlerror());
    return EXIT_FAILURE;
  }
  func3();
#endif

  dlerror();
  void (*gcov_reset1)() = (void (*)())dlsym(f1_handle, "__gcov_reset");
  if (gcov_reset1 == NULL) {
    fprintf(stderr, "unable to find __gcov_reset in func.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  dlerror();
  void (*gcov_reset2)() = (void (*)())dlsym(f2_handle, "__gcov_reset");
  if (gcov_reset2 == NULL) {
    fprintf(stderr, "unable to find __gcov_reset in func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  if (gcov_reset1 == gcov_reset2) {
    fprintf(stderr, "Same __gcov_reset found in func.shared and func2.shared\n");
    return EXIT_FAILURE;
  }

  dlerror();
  if (dlclose(f2_handle) != 0) {
    fprintf(stderr, "unable to close 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  func();

  int g1 = 0;
  int g2 = 0;
  int n = 10;

  if (n % 5 == 0)
    g1++;
  else
    g2++;

  return EXIT_SUCCESS;
}

