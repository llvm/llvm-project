// Set these requirements to ensure that we have an 8-byte binary ID.
// REQUIRES: linux
//
// This will generate a 20-byte build ID, which requires 4-byte padding.
// RUN: %clang_profgen -Wl,--build-id=sha1 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: %run %t %t.profraw

#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

enum ValueKind {
#define VALUE_PROF_KIND(Enumerator, Value, Descr) Enumerator = Value,
#include "profile/InstrProfData.inc"
};

typedef struct __llvm_profile_header {
#define INSTR_PROF_RAW_HEADER(Type, Name, Initializer) Type Name;
#include "profile/InstrProfData.inc"
} __llvm_profile_header;

typedef void *IntPtrT;
typedef struct __llvm_profile_data {
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) Type Name;
#include "profile/InstrProfData.inc"
} __llvm_profile_data;

void bail(const char* str) {
  fprintf(stderr, "%s %s\n", str, strerror(errno));
  exit(1);
}

void func() {}

int main(int argc, char** argv) {
  if (argc == 2) {
    int fd = open(argv[1], O_RDONLY);
    if (fd == -1)
      bail("open");

    struct stat st;
    if (stat(argv[1], &st))
      bail("stat");
    uint64_t FileSize = st.st_size;

    char* Buf = (char *)mmap(NULL, FileSize, PROT_READ, MAP_SHARED, fd, 0);
    if (Buf == MAP_FAILED)
      bail("mmap");

    __llvm_profile_header *Header = (__llvm_profile_header *)Buf;
    if (Header->BinaryIdsSize != 32)
      bail("Invalid binary ID size");

    char *BinaryIdsStart = Buf + sizeof(__llvm_profile_header);

    uint64_t BinaryIdSize = *((uint64_t *)BinaryIdsStart);
    if (BinaryIdSize != 20)
      bail("Expected a binary ID of size 20");

    // Skip the size and the binary ID itself to check padding.
    BinaryIdsStart += 8 + 20;
    if (*((uint32_t *)BinaryIdsStart))
      bail("Found non-zero binary ID padding");

    if (munmap(Buf, FileSize))
      bail("munmap");

    if (close(fd))
      bail("close");
  } else {
    func();
  }
  return 0;
}
