// The identifier table only contains the builtins that get used, not all of
// the ~12,000 builtins of an AArch64 target. An empty file needs fewer than
// 2,000 identifiers (keywords and the like).

// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -print-stats %s \
// RUN:   2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -print-stats \
// RUN:   -x c++ %s 2>&1 | FileCheck %s

// CHECK: *** Identifier Table Stats:
// CHECK-NEXT: # Identifiers: {{ *1?[0-9]{1,3}$}}
