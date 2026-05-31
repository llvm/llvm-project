

#ifndef PROFILE_TEST_H
#define PROFILE_TEST_H

#if defined(_MSC_VER)
# define ALIGNED(x) __declspec(align(x))
#else  // _MSC_VER
# define ALIGNED(x) __attribute__((aligned(x)))
#endif

inline void __llvm_profile_test_initialize() {
  // This is a no-op on most platforms, but on AIX it forces the linker to
  // keep the start/stop stub data for the runtime. Normally this data is
  // referenced by pulling in `__llvm_profile_runtime` from the runtime but
  // some tests explicitly supress that.
#ifdef _AIX
  extern __attribute__((visibility("hidden"))) void *__llvm_profile_keep[];
  (void)*(void *volatile *)__llvm_profile_keep;
#endif // _AIX
}
#endif  // PROFILE_TEST_H
