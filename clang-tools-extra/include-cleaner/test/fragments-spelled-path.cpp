#include <vector>
#include "generated/gen.inc"

SpelledGen G;

// RUN: clang-include-cleaner -print=changes %s --fragment-headers='generated/gen\.inc' --fragment-dependency-comment-format='needed by {0}' -- -I%S/Inputs/ | FileCheck --check-prefix=CHANGES %s
// CHANGES: ~ <vector> @Line:1 // needed by "generated/gen.inc"
// CHANGES-NOT: - <vector>
