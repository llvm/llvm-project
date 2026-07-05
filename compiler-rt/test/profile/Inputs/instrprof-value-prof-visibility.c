#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef DLOPEN_FUNC_DIR
#include <dlfcn.h>
#endif

int __llvm_profile_runtime = 0;
int __llvm_profile_write_file();
void __llvm_profile_reset_counters(void);
void __llvm_profile_initialize_file(void);
struct __llvm_profile_data;
struct ValueProfData;
void lprofMergeValueProfData(struct ValueProfData *, struct __llvm_profile_data *);
/* Force the vp merger module to be linked in.  */
void *Dummy = &lprofMergeValueProfData;

void callee1() {}
void callee2() {}

typedef void (*FP)(void);
FP Fps[2] = {callee1, callee2};

int main(int argc, char *argv[]) {
  __llvm_profile_initialize_file();
  __llvm_profile_write_file();
  __llvm_profile_reset_counters();

#ifdef DLOPEN_FUNC_DIR
  void *Handle = dlopen(DLOPEN_FUNC_DIR "/func.shared", RTLD_NOW);
  if (!Handle) {
    fprintf(stderr, "unable to open '" DLOPEN_FUNC_DIR "/func.shared': %s\n",
            dlerror());
    return EXIT_FAILURE;
  }

  // This tests that lprofMergeValueProfData is not accessed
  // from outside a module
  void (*SymHandle)(struct ValueProfData *, struct __llvm_profile_data *) =
      (void (*)(struct ValueProfData *, struct __llvm_profile_data *))dlsym(
          Handle, "lprofMergeValueProfData");
  if (SymHandle) {
    fprintf(stderr,
            "should not be able to lookup symbol 'lprofMergeValueProfData': %s\n",
            dlerror());
    return EXIT_FAILURE;
  }

  dlclose(Handle);

#endif

  Fps[0]();
  Fps[1]();

  __llvm_profile_write_file();
  __llvm_profile_reset_counters();

  return EXIT_SUCCESS;
}
