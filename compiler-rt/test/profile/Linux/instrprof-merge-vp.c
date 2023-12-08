// RUN: %clang_profgen -mllvm --enable-value-profiling=true -mllvm -vp-static-alloc=true -mllvm -vp-counters-per-site=3 -O2 -o %t %s
// RUN: %run %t %t.profraw
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --all-functions --counts --ic-targets %t.profdata > %t.profdump
// RUN: FileCheck --input-file %t.profdump  %s --check-prefix=FOO
// RUN: FileCheck --input-file %t.profdump  %s --check-prefix=BAR

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

int __llvm_profile_runtime = 0;
int __llvm_profile_write_file();
void __llvm_profile_reset_counters(void);
int __llvm_profile_merge_from_buffer(const char *, uint64_t);
void __llvm_profile_set_filename(const char *);
struct __llvm_profile_data;
struct ValueProfData;
void lprofMergeValueProfData(struct ValueProfData *, struct __llvm_profile_data *);
/* Force the vp merger module to be linked in.  */
void *Dummy = &lprofMergeValueProfData;

void callee1() {}
void callee2() {}
void callee3() {}

typedef void (*FP)(void);
FP Fps[3] = {callee1, callee2, callee3};

void foo(int N) {
  int I, J;
  for (I = 0; I < 3; I++)
    for (J = 0; J < I * 2 + 1; J++)
      Fps[I]();

  if (N < 2)
    return;

  for (I = 0; I < 3; I++)
    for (J = 0; J < I * 2 + 1; J++)
      Fps[2 - I]();
}

/* This function is not profiled */
void bar(void) {
  int I;
  for (I = 0; I < 20; I++)
    Fps[I % 3]();
}

int main(int argc, const char *argv[]) {
  int i;
  if (argc < 2)
    return 1;

  const char *FileN = argv[1];
  __llvm_profile_set_filename(FileN);
  /* Start profiling. */
  __llvm_profile_reset_counters();
  foo(1);
  /* End profiling by freezing counters and
   * dump them to the file. */
  if (__llvm_profile_write_file())
    return 1;

  /* Read profile data into buffer. */
  FILE *File = fopen(FileN, "r");
  if (!File)
    return 1;
  fseek(File, 0, SEEK_END);
  uint64_t Size = ftell(File);
  fseek(File, 0, SEEK_SET);
  char *Buffer = (char *)malloc(Size);
  if (Size != fread(Buffer, 1, Size, File))
    return 1;
  fclose(File);

  /* Its profile will be discarded. */
  for (i = 0; i < 10; i++)
    bar();

  /* Start profiling again and merge in previously
     saved counters in buffer. */
  __llvm_profile_reset_counters();
  __llvm_profile_merge_from_buffer(Buffer, Size);
  foo(2);
  /* End profiling. */
  truncate(FileN, 0);
  if (__llvm_profile_write_file())
    return 1;

  /* Its profile will be discarded. */
  bar();

  return 0;
}

// FOO-LABEL:  foo:
// FOO:    Indirect Target Results:
// FOO-NEXT:	[ 0, callee3, 10 ]
// FOO-NEXT:	[ 0, callee2, 6 ]
// FOO-NEXT:	[ 0, callee1, 2 ]
// FOO-NEXT:	[ 1, callee1, 5 ]
// FOO-NEXT:	[ 1, callee2, 3 ]
// FOO-NEXT:	[ 1, callee3, 1 ]

// BAR-LABEL: bar:
// BAR:         [ 0, callee1, 0 ]
// BAR-NEXT:    [ 0, callee2, 0 ]
// BAR-NEXT:    [ 0, callee3, 0 ]

