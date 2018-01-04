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

  void *f2_handle = dlopen("func2.shared", RTLD_LAZY | RTLD_GLOBAL);
  if (f2_handle == NULL) {
    fprintf(stderr, "unable to open 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  if (dlclose(f2_handle) != 0) {
    fprintf(stderr, "unable to close 'func2.shared': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

