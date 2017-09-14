// RUN: not clang-refactor 2>&1 | FileCheck --check-prefix=MISSING_ACTION %s
// MISSING_ACTION: error: no refactoring action given
// MISSING_ACTION-NEXT: note: the following actions are supported:

// RUN: not clang-refactor local-rename -no-dbs 2>&1 | FileCheck --check-prefix=MISSING_SOURCES %s
// MISSING_SOURCES: error: must provide paths to the source files when '-no-dbs' is used
