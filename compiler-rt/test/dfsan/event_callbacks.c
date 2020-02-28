// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -c %s -o %t-callbacks.o
// RUN: %clang_dfsan -O2 -mllvm -dfsan-event-callbacks %s %t-callbacks.o -o %t
// RUN: %run %t FooBarBaz 2>&1 | FileCheck %s

// Tests that callbacks are inserted for store events when
// -dfsan-event-callbacks is specified.

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdio.h>
#include <string.h>

#ifdef CALLBACKS
// Compile this code without DFSan to avoid recursive instrumentation.

extern dfsan_label LabelI;
extern dfsan_label LabelJ;
extern dfsan_label LabelIJ;
extern dfsan_label LabelArgv;
extern size_t LenArgv;

void __dfsan_store_callback(dfsan_label Label) {
  if (!Label)
    return;

  static int Count = 0;
  switch (Count++) {
  case 0:
    assert(Label == LabelI);
    break;
  case 1:
    assert(Label == LabelJ);
    break;
  case 2:
    assert(Label == LabelIJ);
    break;
  default:
    assert(0);
  }

  fprintf(stderr, "Label %u stored to memory\n", Label);
}

void __dfsan_load_callback(dfsan_label Label) {
  if (!Label)
    return;

  fprintf(stderr, "Label %u loaded from memory\n", Label);
}

void __dfsan_mem_transfer_callback(dfsan_label *Start, size_t Len) {
  assert(Len == LenArgv);
  for (int I = 0; I < Len; ++I) {
    assert(Start[I] == LabelArgv);
  }

  fprintf(stderr, "Label %u copied to memory\n", Start[0]);
}

#else
// Compile this code with DFSan and -dfsan-event-callbacks to insert the
// callbacks.

dfsan_label LabelI;
dfsan_label LabelJ;
dfsan_label LabelIJ;
dfsan_label LabelArgv;

size_t LenArgv;

int main(int Argc, char *Argv[]) {
  assert(Argc == 2);

  int I = 1, J = 2;
  LabelI = dfsan_create_label("I", 0);
  dfsan_set_label(LabelI, &I, sizeof(I));
  LabelJ = dfsan_create_label("J", 0);
  dfsan_set_label(LabelJ, &J, sizeof(J));
  LabelIJ = dfsan_union(LabelI, LabelJ);

  // CHECK: Label 1 stored to memory
  volatile int Sink = I;

  // CHECK: Label 1 loaded from memory
  assert(Sink == 1);

  // CHECK: Label 2 stored to memory
  Sink = J;

  // CHECK: Label 2 loaded from memory
  assert(Sink == 2);

  // CHECK: Label 2 loaded from memory
  // CHECK: Label 3 stored to memory
  Sink += I;

  // CHECK: Label 3 loaded from memory
  assert(Sink == 3);

  LenArgv = strlen(Argv[1]);
  LabelArgv = dfsan_create_label("Argv", 0);
  dfsan_set_label(LabelArgv, Argv[1], LenArgv);

  char SinkBuf[64];
  assert(LenArgv < sizeof(SinkBuf) - 1);

  // CHECK: Label 4 copied to memory
  memcpy(SinkBuf, Argv[1], LenArgv);

  // CHECK: Label 4 copied to memory
  memmove(&SinkBuf[1], SinkBuf, LenArgv);

  return 0;
}

#endif // #ifdef CALLBACKS
