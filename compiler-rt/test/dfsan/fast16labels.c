// RUN: %clang_dfsan %s -o %t
// RUN: DFSAN_OPTIONS=fast16labels=1 %run %t
// RUN: DFSAN_OPTIONS=fast16labels=1 not %run %t dfsan_create_label 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CREATE-LABEL
// RUN: DFSAN_OPTIONS=fast16labels=1 not %run %t dfsan_get_label_info 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GET-LABEL-INFO
// RUN: DFSAN_OPTIONS=fast16labels=1 not %run %t dfsan_has_label_with_desc \
// RUN:   2>&1 | FileCheck %s --check-prefix=HAS-LABEL-WITH-DESC
// RUN: DFSAN_OPTIONS=fast16labels=1:dump_labels_at_exit=/dev/stdout not %run \
// RUN:   %t 2>&1 | FileCheck %s --check-prefix=DUMP-LABELS
//
// Tests DFSAN_OPTIONS=fast16labels=1
//
#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

int foo(int a, int b) {
  return a + b;
}

int main(int argc, char *argv[]) {
  // Death tests for unsupported API usage.
  const char *command = (argc < 2) ? "" : argv[1];
  // CREATE-LABEL: FATAL: DataFlowSanitizer: dfsan_create_label is unsupported
  if (strcmp(command, "dfsan_create_label") == 0)
    dfsan_create_label("", NULL);
  // GET-LABEL-INFO: FATAL: DataFlowSanitizer: dfsan_get_label_info is unsupported
  if (strcmp(command, "dfsan_get_label_info") == 0)
    dfsan_get_label_info(1);
  // HAS-LABEL-WITH-DESC: FATAL: DataFlowSanitizer: dfsan_has_label_with_desc is unsupported
  if (strcmp(command, "dfsan_has_label_with_desc") == 0)
    dfsan_has_label_with_desc(1, "");
  // DUMP-LABELS: FATAL: DataFlowSanitizer: dfsan_dump_labels is unsupported

  // Supported usage.
  int a = 10;
  int b = 20;
  dfsan_set_label(8, &a, sizeof(a));
  dfsan_set_label(512, &b, sizeof(b));
  int c = foo(a, b);
  printf("A: 0x%x\n", dfsan_get_label(a));
  printf("B: 0x%x\n", dfsan_get_label(b));
  dfsan_label l = dfsan_get_label(c);
  printf("C: 0x%x\n", l);
  assert(l == 520);  // OR of the other two labels.
  assert(dfsan_has_label(l, 8));
  assert(dfsan_has_label(l, 512));
  assert(!dfsan_has_label(l, 1));
}
