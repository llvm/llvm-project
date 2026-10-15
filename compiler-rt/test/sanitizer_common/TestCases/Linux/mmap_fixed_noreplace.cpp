// RUN: %clangxx -O1 %s -o %t && %run %t
// REQUIRES: x86_64-target-arch
// UNSUPPORTED: lsan, ubsan
// FIXME: Support MAP_FIXED_NOREPLACE in other sanitizers.
// XFAIL: asan, hwasan, msan

#include <assert.h>
#include <errno.h>
#include <initializer_list>
#include <stdint.h>
#include <sys/mman.h>

#if __has_feature(hwaddress_sanitizer)
extern "C" uintptr_t __hwasan_shadow_memory_dynamic_address;
#endif

int main() {
  // A canonical Linux x86-64 address outside sanitizer application memory.
#if __has_feature(hwaddress_sanitizer)
  void *address =
      reinterpret_cast<void *>(__hwasan_shadow_memory_dynamic_address);
#else
  void *address = reinterpret_cast<void *>(0x080000000000ULL);
#endif
  const int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  for (int fixed : {MAP_FIXED, MAP_FIXED_NOREPLACE}) {
    errno = 0;
    void *result = mmap(address, 4096, PROT_NONE, flags | fixed, -1, 0);
    assert(result == MAP_FAILED);
    assert(errno == EINVAL);
  }

  // An ordinary hint may still be changed to a supported application address.
  void *result = mmap(address, 4096, PROT_NONE, flags, -1, 0);
  assert(result != MAP_FAILED);
  assert(result != address);
  assert(munmap(result, 4096) == 0);
  return 0;
}
