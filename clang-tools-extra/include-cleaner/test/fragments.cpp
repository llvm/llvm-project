#include <vector>
#include "gen.inc"

Gen G;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers=.*\\.inc$ --fragment-dependency-comment-format='needed by {0}' -- -I%S/Inputs/ | FileCheck --check-prefix=CHANGES %s
// CHANGES: ~ <vector> @Line:1 // needed by "gen.inc"
// CHANGES-NOT: - <vector>

// RUN: clang-include-cleaner -print %s --fragment-headers=.*\\.inc$ --fragment-dependency-comment-format='needed by {0}' -- -I%S/Inputs/ | FileCheck --check-prefix=PRINT %s
// PRINT: #include <vector> // needed by "gen.inc"

// RUN: cp %s %t.cpp
// RUN: clang-include-cleaner -edit %t.cpp --fragment-headers=.*\\.inc$ --fragment-dependency-comment-format='IWYU pragma: keep' -- -I%S/Inputs/
// RUN: FileCheck --check-prefix=EDIT %s < %t.cpp
// EDIT: #include <vector> // IWYU pragma: keep
