// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wsystem-headers \
// RUN:   -fretain-comments-from-system-headers -isystem %S/Inputs \
// RUN:   -include documentation-system-header-doc.h %s 2>&1 | FileCheck %s

// Comments in a system header normally go unchecked, because the warnings are
// off there. -Wsystem-headers turns them back on, and then the comment does
// have to be checked -- so the answer cannot be hardcoded for system headers.

// CHECK: documentation-system-header-doc.h:1:6: warning: '\returns' command used in a comment that is attached to a function returning void

void user_fn();
