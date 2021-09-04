// REQUIRES: native-run
// UNSUPPORTED: arm, aarch64
// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_clear_cache

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

extern void __clear_cache(void* start, void* end);

typedef int (*pfunc)(void);

// Make these static to avoid ILT jumps for incremental linking on Windows.
static int func1() { return 1; }
static int func2() { return 2; }

void __attribute__((noinline))
memcpy_f(void *dst, const void *src, size_t n) {
// ARM and MIPS naturally align functions, but use the LSB for ISA selection
// (THUMB, MIPS16/uMIPS respectively).  Ensure that the ISA bit is ignored in
// the memcpy
#if defined(__arm__) || defined(__mips__)
  memcpy(dst, (void *)((uintptr_t)src & ~1), n);
#else
  memcpy(dst, (void *)((uintptr_t)src), n);
#endif
}

// Realign the 'dst' pointer as if it has been returned by memcpy() above.
// We need to split it because we're using two mappings for the same area.
void *__attribute__((noinline))
realign_f(void *dst, const void *src, size_t n) {
#if defined(__arm__) || defined(__mips__)
  return (void *)((uintptr_t)dst | ((uintptr_t)src & 1));
#else
  return dst;
#endif
}

int main()
{
    const int kSize = 128;
#if defined(__NetBSD__)
    // we need to create separate RW and RX mappings to satisfy MPROTECT
    uint8_t *write_buffer = mmap(0, kSize,
                                 PROT_MPROTECT(PROT_READ | PROT_WRITE |
                                               PROT_EXEC),
                                 MAP_ANON | MAP_PRIVATE, -1, 0);
    if (write_buffer == MAP_FAILED)
      return 1;
    uint8_t *execution_buffer = mremap(write_buffer, kSize, NULL, kSize,
                                       MAP_REMAPDUP);
    if (execution_buffer == MAP_FAILED)
      return 1;

    if (mprotect(write_buffer, kSize, PROT_READ | PROT_WRITE) == -1)
      return 1;
    if (mprotect(execution_buffer, kSize, PROT_READ | PROT_EXEC) == -1)
      return 1;
#elif !defined(_WIN32)
    uint8_t *execution_buffer = mmap(0, kSize,
                                     PROT_READ | PROT_WRITE | PROT_EXEC,
                                     MAP_ANON | MAP_PRIVATE, -1, 0);
    if (execution_buffer == MAP_FAILED)
      return 1;
    uint8_t *write_buffer = execution_buffer;
#else
    HANDLE mapping = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                       PAGE_EXECUTE_READWRITE, 0, kSize, NULL);
    if (mapping == NULL)
        return 1;

    uint8_t* execution_buffer = MapViewOfFile(
        mapping, FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE, 0, 0, 0);
    if (execution_buffer == NULL)
        return 1;
    uint8_t *write_buffer = execution_buffer;
#endif

    // verify you can copy and execute a function
    memcpy_f(write_buffer, func1, kSize);
    pfunc f1 = (pfunc)realign_f(execution_buffer, func1, kSize);
    __clear_cache(execution_buffer, execution_buffer + kSize);
    if ((*f1)() != 1)
        return 1;

    // verify you can overwrite a function with another
    memcpy_f(write_buffer, func2, kSize);
    pfunc f2 = (pfunc)realign_f(execution_buffer, func2, kSize);
    __clear_cache(execution_buffer, execution_buffer + kSize);
    if ((*f2)() != 2)
        return 1;

    return 0;
}
