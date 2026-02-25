// RUN: cd %S
// RUN: %clang -MM -MP -I Inputs -Xclang -fdepfile-entry=1.extra -Xclang -fdepfile-entry=2.extra -Xclang -fdepfile-entry=2.extra dependency-gen-phony.c | \
// RUN:   FileCheck %s %if system-darwin && target={{.*}}-{{darwin|macos}}{{.*}} %{ --check-prefix=CHECK-DARWIN %} --match-full-lines --strict-whitespace --implicit-check-not=.c:
// RUN: %clang -MM -MP -I Inputs -xc - < dependency-gen-phony.c | FileCheck %s --check-prefix=%if system-darwin && target={{.*}}-{{darwin|macos}}{{.*}} %{STDIO-DARWIN %} %else %{STDIO %} --implicit-check-not=.c:

/// Verify that phony targets are only created for the extra dependency files,
/// and not the input file.
//       CHECK:dependency-gen-phony.o: 1.extra 2.extra dependency-gen-phony.c \
//  CHECK-NEXT:  Inputs{{/|\\}}empty.h
//  CHECK-NEXT:1.extra:
//  CHECK-NEXT:2.extra:
//  CHECK-NEXT:{{.*}}empty.h:
//   CHECK-NOT:{{.}}

//       CHECK-DARWIN:dependency-gen-phony.o: \
//  CHECK-DARWIN-NEXT:  {{.*}}SDKSettings.json \
//  CHECK-DARWIN-NEXT:  1.extra 2.extra dependency-gen-phony.c Inputs{{/|\\}}empty.h
//  CHECK-DARWIN-NEXT:{{.*}}SDKSettings.json:
//  CHECK-DARWIN-NEXT:1.extra:
//  CHECK-DARWIN-NEXT:2.extra:
//  CHECK-DARWIN-NEXT:{{.*}}empty.h:
//   CHECK-DARWIN-NOT:{{.}}

// STDIO:      -.o: Inputs{{/|\\}}empty.h
// STDIO-NEXT: {{.*}}empty.h:
// STDIO-NOT:  {{.}}

// STDIO-DARWIN:      -.o:{{.*[[:space:]].*}}SDKSettings.json
// STDIO-DARWIN-NEXT: Inputs{{/|\\}}empty.h
// STDIO-DARWIN-NEXT: {{.*}}SDKSettings.json:
// STDIO-DARWIN-NEXT: {{.*}}empty.h:
// STDIO-DARWIN-NOT:  {{.}}

#include "empty.h"
