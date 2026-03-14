
#include <stdio.h>
#include <stdlib.h>

#ifdef DLOPEN_FUNC_DIR
#include <dlfcn.h>
#else
void func(int K);
void func2(int K);
#endif

int main(int argc, char *argv[]) {
#ifdef DLOPEN_FUNC_DIR
  dlerror();
  void *f1_handle = dlopen(DLOPEN_FUNC_DIR"/func.shared", DLOPEN_FLAGS);
  if (f1_handle == NULL) {
    fprintf(stderr, "unable to open '" DLOPEN_FUNC_DIR "/func.shared': %s\n",
            dlerror());
    return EXIT_FAILURE;
  }

  void (*func)(int) = (void (*)(int))dlsym(f1_handle, "func");
  if (func == NULL) {
    fprintf(stderr, "unable to lookup symbol 'func': %s\n", dlerror());
    return EXIT_FAILURE;
  }

  void *f2_handle = dlopen(DLOPEN_FUNC_DIR"/func2.shared", DLOPEN_FLAGS);
  if (f2_handle == NULL) {
    fprintf(stderr, "unable to open '" DLOPEN_FUNC_DIR "/func2.shared': %s\n",
            dlerror());
    return EXIT_FAILURE;
  }

  void (*func2)(int) = (void (*)(int))dlsym(f2_handle, "func2");
  if (func2 == NULL) {
    fprintf(stderr, "unable to lookup symbol 'func2': %s\n", dlerror());
    return EXIT_FAILURE;
  }
#endif

  func(1);
  func2(0);

  return EXIT_SUCCESS;
}

