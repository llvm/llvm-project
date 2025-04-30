/* When building with GCC with static-only libgcc, the dummy
   _Unwind_Resume from static-stubs.c needs to be used together with
   the dummy __aeabi_unwind_cpp_pr* from aeabi_unwind_cpp_pr1.c
   instead of the copies from libgcc.  */

#include <elf/static-stubs.c>
#include <aeabi_unwind_cpp_pr1.c>
