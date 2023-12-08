#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int __llvm_profile_runtime = 0;
uint64_t __llvm_profile_get_size_for_buffer(void);
int __llvm_profile_write_buffer(char *);
void __llvm_profile_reset_counters(void);
int __llvm_profile_check_compatibility(const char *, uint64_t);

int gg = 0;
void bar(char c) {
  if (c == '1')
    gg++;
  else
    gg--;
}

/* Returns 0 (size) when an error occurs. */
uint64_t libEntry(char *Buffer, uint64_t MaxSize) {

  uint64_t Size = __llvm_profile_get_size_for_buffer();
  if (Size > MaxSize)
    return 0;

  __llvm_profile_reset_counters();

  bar('1');

  if (__llvm_profile_write_buffer(Buffer))
    return 0;

  /* Now check compatibility. Should return 0.  */
  if (__llvm_profile_check_compatibility(Buffer, Size))
    return 0;

  return Size;
}

