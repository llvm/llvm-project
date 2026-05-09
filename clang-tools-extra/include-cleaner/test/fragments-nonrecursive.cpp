#include <vector>
#include "outer.inc"

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='.*\.inc$' -- -I%S/Inputs/ | FileCheck --check-prefix=CHANGES %s
// CHANGES: - <vector> @Line:1
