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

  dlerror();
  void *f2_handle = dlopen("func2.shared", RTLD_LAZY | RTLD_GLOBAL);
  if (f2_handle == NULL) {
    fprintf(stderr, "unable to open 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  dlerror();
  void (*gcov_flush)() = (void (*)())dlsym(f1_handle, "__gcov_flush");
  if (gcov_flush != NULL || dlerror() == NULL) {
    fprintf(stderr, "__gcov_flush should not be visible in func.shared'\n");
    return EXIT_FAILURE;
  }

  dlerror();
  void (*f1_flush)() = (void (*)())dlsym(f1_handle, "llvm_gcov_flush");
  if (f1_flush == NULL) {
    fprintf(stderr, "unable to find llvm_gcov_flush in func.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }
  f1_flush();

  dlerror();
  void (*f2_flush)() = (void (*)())dlsym(f2_handle, "llvm_gcov_flush");
  if (f2_flush == NULL) {
    fprintf(stderr, "unable to find llvm_gcov_flush in func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }
  f2_flush();

  if (f1_flush == f2_flush) {
    fprintf(stderr, "Same llvm_gcov_flush found in func.shared and func2.shared\n");
    return EXIT_FAILURE;
  }

  dlerror();
  if (dlclose(f2_handle) != 0) {
    fprintf(stderr, "unable to close 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

