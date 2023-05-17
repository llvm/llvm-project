// RUN: cd %S
// RUN: %clang -MM -MP -I Inputs -Xclang -fdepfile-entry=1.extra -Xclang -fdepfile-entry=2.extra -Xclang -fdepfile-entry=2.extra dependency-gen-phony.c | \
// RUN:   FileCheck %s --match-full-lines --strict-whitespace --implicit-check-not=.c:
// RUN: %clang -MM -MP -I Inputs -xc - < dependency-gen-phony.c | FileCheck %s --check-prefix=STDIO --implicit-check-not=.c:

/// Verify that phony targets are only created for the extra dependency files,
/// and not the input file.
//       CHECK:dependency-gen-phony.o: 1.extra 2.extra dependency-gen-phony.c \
//  CHECK-NEXT:  Inputs{{/|\\}}empty.h
//  CHECK-NEXT:1.extra:
//  CHECK-NEXT:2.extra:
//  CHECK-NEXT:{{.*}}empty.h:
//   CHECK-NOT:{{.}}

// STDIO:      -.o: Inputs{{/|\\}}empty.h
// STDIO-NEXT: {{.*}}empty.h:
// STDIO-NOT:  {{.}}

#include "empty.h"
