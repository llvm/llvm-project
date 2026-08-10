// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -c %s -o %t-callbacks.o
// RUN: %clang_dfsan -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -O2 -mllvm -dfsan-conditional-callbacks %s %t-callbacks.o -o %t
// RUN: %run %t FooBarBaz 2>&1 | FileCheck %s
//
// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -DORIGINS -c %s -o %t-callbacks-orig.o
// RUN: %clang_dfsan -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -O2 -mllvm -dfsan-conditional-callbacks -mllvm -dfsan-track-origins=1 -DORIGINS %s %t-callbacks-orig.o -o %t-orig
// RUN: %run %t-orig FooBarBaz 2>&1 | FileCheck %s

// Tests that callbacks are inserted for conditionals when
// -dfsan-conditional-callbacks is specified.

#include <assert.h>
#include <sanitizer/dfsan_interface.h>
#include <stdio.h>
#include <string.h>

#ifdef CALLBACKS
// Compile this code without DFSan to avoid recursive instrumentation.

extern dfsan_label LabelI;
extern dfsan_label LabelJ;
extern dfsan_label LabelIJ;

void my_dfsan_conditional_callback(dfsan_label Label, dfsan_origin Origin) {
  assert(Label != 0);
#ifdef ORIGINS
  assert(Origin != 0);
#else
  assert(Origin == 0);
#endif

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
    break;
  }

  fprintf(stderr, "Label %u used as condition\n", Label);
}

#else
// Compile this code with DFSan and -dfsan-conditional-callbacks to insert the
// callbacks.

dfsan_label LabelI;
dfsan_label LabelJ;
dfsan_label LabelIJ;

extern void my_dfsan_conditional_callback(dfsan_label Label,
                                          dfsan_origin Origin);

int main(int Argc, char *Argv[]) {
  assert(Argc == 2);

  dfsan_set_conditional_callback(my_dfsan_conditional_callback);

  int result = 0;
  // Make these not look like constants, otherwise the branch we're expecting
  // may be optimized out.
  int DataI = (Argv[0][0] != 0) ? 1 : 0;
  int DataJ = (Argv[1][0] != 0) ? 2 : 0;
  LabelI = 1;
  dfsan_set_label(LabelI, &DataI, sizeof(DataI));
  LabelJ = 2;
  dfsan_set_label(LabelJ, &DataJ, sizeof(DataJ));
  LabelIJ = dfsan_union(LabelI, LabelJ);

  assert(dfsan_get_label(DataI) == LabelI);

  // CHECK: Label 1 used as condition
  if (DataI) {
    result = 42;
  }
  
  fprintf(stderr, "Result is %d\n", result);
  assert(dfsan_get_label(DataJ) == LabelJ);

  // CHECK: Label 2 used as condition
  switch (DataJ) {
  case 1:
    result += 10000;
    break;
  case 2:
    result += 4200;
    break;
  default:
    break;
  }

  int tainted_cond = ((DataI * DataJ) != 1);
  fprintf(stderr, "Result is %d\n", result);
  assert(dfsan_get_label(tainted_cond) == LabelIJ);

  // CHECK: Label 3 used as condition
  result = tainted_cond ? result + 420000 : 9;

  fprintf(stderr, "Result is %d\n", result);
  assert(result == 424242);
  return 0;
}

#endif // #ifdef CALLBACKS
