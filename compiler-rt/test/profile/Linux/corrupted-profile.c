// RUN: rm -f %t.profraw
// RUN: touch %t.profraw
// RUN: %clang_profgen -o %t %s
// RUN: %t %t.profraw
// RUN: %t %t.profraw modifyfile
// RUN: cp %t.profraw %t.profraw.old
// RUN: %t %t.profraw 2>&1 | FileCheck %s
// RUN: diff %t.profraw %t.profraw.old
// CHECK: Invalid profile data to merge

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

void __llvm_profile_set_file_object(FILE* File, int EnableMerge);

void bail(const char* str) {
  fprintf(stderr, "%s %s\n", str, strerror(errno));
  exit(1);
}

int main(int argc, char** argv) {
  if (argc == 3) {
    int fd = open(argv[1], O_RDWR);
    if (fd == -1)
      bail("open");

    struct stat st;
    if (stat(argv[1], &st))
      bail("stat");
    uint64_t FileSize = st.st_size;

    char* Buf = (char *) mmap(NULL, FileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (Buf == MAP_FAILED)
      bail("mmap");

    __llvm_profile_header *Header = (__llvm_profile_header *)Buf;
    __llvm_profile_data *SrcDataStart =
        (__llvm_profile_data *)(Buf + sizeof(__llvm_profile_header) +
                                Header->BinaryIdsSize);
    memset(&SrcDataStart->CounterPtr, 0xAB, sizeof(SrcDataStart->CounterPtr));

    if (munmap(Buf, FileSize))
      bail("munmap");

    if (close(fd))
      bail("close");
  } else {
    FILE* f = fopen(argv[1], "r+b");
    if (!f)
      bail("fopen");
    __llvm_profile_set_file_object(f, 1);
  }
}
