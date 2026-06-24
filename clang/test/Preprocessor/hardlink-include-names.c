// Test that symlinked and hardlinked files are reported with their correct
// filenames
//
// REQUIRES: symlinks
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: ln %t/foo.inc %t/bar.inc
// RUN: ln -s %t/foo.inc %t/baz.inc

// Test 1: -E should show both filenames.
// RUN: %clang_cc1 -E %t/main.c -o - | FileCheck --check-prefix=PP %s
// PP: # 1 "{{.*(/|\\\\)}}foo.inc" 1
// PP: # 1 "{{.*(/|\\\\)}}bar.inc" 1
// PP: # 1 "{{.*(/|\\\\)}}baz.inc" 1

// Test 2: .d should list both filenames.
// RUN: %clang_cc1 -dependency-file %t/deps.d -MT main.o %t/main.c -fsyntax-only
// RUN: FileCheck --check-prefix=DEPS -input-file=%t/deps.d %s
// DEPS: foo.inc
// DEPS: bar.inc
// DEPS: baz.inc

// Test 3: --show-includes should list both filenames.
// RUN: %clang_cc1 --show-includes -o /dev/null %t/main.c | \
// RUN:   FileCheck --check-prefix=SHOW %s
// SHOW: Note: including file: {{.*}}foo.inc
// SHOW: Note: including file: {{.*}}bar.inc
// SHOW: Note: including file: {{.*}}baz.inc

//--- main.c
const char *a =
#include "foo.inc"
;
const char *b =
#include "bar.inc"
;
const char *c =
#include "baz.inc"
;

//--- foo.inc
"contents"
