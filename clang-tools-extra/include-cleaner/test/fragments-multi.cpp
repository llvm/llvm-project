#include <vector>
#include "a.inc"
#include "b.inc"

A AValue;
B BValue;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='.*\.inc$' --fragment-dependency-comment-format='needed by {0}' -- -I%S/Inputs/ | FileCheck --check-prefix=CHANGES %s
// CHANGES: ~ <vector> @Line:1 // needed by "a.inc", "b.inc"
// CHANGES-NOT: - <vector>
