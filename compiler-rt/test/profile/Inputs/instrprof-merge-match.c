#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../profile_test.h"

int __llvm_profile_runtime = 0;
uint64_t __llvm_profile_get_size_for_buffer(void);
int __llvm_profile_write_buffer(char *);
void __llvm_profile_reset_counters(void);
int  __llvm_profile_check_compatibility(const char *, uint64_t);

int g = 0;
void foo(char c) {
  if (c == '1')
    g++;
  else
    g--;
}

extern uint64_t libEntry(char *Buffer, uint64_t MaxSize);

int main(int argc, const char *argv[]) {
  const uint64_t MaxSize = 10000;
  static char ALIGNED(sizeof(uint64_t)) Buffer[MaxSize];

  uint64_t Size = __llvm_profile_get_size_for_buffer();
  if (Size > MaxSize)
    return 1;

  __llvm_profile_reset_counters();
  foo('0');

  if (__llvm_profile_write_buffer(Buffer))
    return 1;

  /* Now check compatibility. Should return 0.  */
  if (__llvm_profile_check_compatibility(Buffer, Size))
    return 1;

  /* Clear the buffer. */
  memset(Buffer, 0, MaxSize);

  /* Collect profile from shared library.  */
  Size = libEntry(Buffer, MaxSize);

  if (!Size)
    return 1;

  /* Shared library's profile should not match main executable's. */
  if (!__llvm_profile_check_compatibility(Buffer, Size))
    return 1;

  return 0;
}

