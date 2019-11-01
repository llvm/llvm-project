// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clang_pgogen -o %t.exe %s
//
// Note: %%p is needed here, not %p, because of lit's path substitution.
// RUN: env LLVM_PROFILE_FILE="%t.dir/%c-%%p" %run %t.exe

#include <stdlib.h>
#include <string.h>

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern const char *__llvm_profile_get_filename(void);
extern int getpid(void);

int main() {
  // Check that continuous mode is enabled.
  if (!__llvm_profile_is_continuous_mode_enabled())
    return 1;

  // Check that the PID is actually in the filename.
  const char *Filename = __llvm_profile_get_filename();

  int Len = strlen(Filename);
  --Len;
  while (Filename[Len] != '-')
    --Len;

  const char *PidStr = Filename + Len + 1;
  int Pid = atoi(PidStr);

  if (Pid != getpid())
    return 1;

  return 0;
}
