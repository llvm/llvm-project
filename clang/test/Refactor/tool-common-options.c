// RUN: not clang-refactor 2>&1 | FileCheck --check-prefix=MISSING_ACTION %s
// MISSING_ACTION: error: no refactoring action given
// MISSING_ACTION-NEXT: note: the following actions are supported:
