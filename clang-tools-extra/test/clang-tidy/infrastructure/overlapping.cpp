// RUN: clang-tidy -checks=-*,llvm-include-order -header-filter=.* %s \
// RUN:   -- -isystem %S/Inputs/Headers -I %S/Inputs/overlapping | \
// RUN:   not grep "note: this fix will not be applied because it overlaps with another fix"

#include <s.h>
#include "o.h"

// Test that clang-tidy takes into account in which file we are doing the
// replacements to determine if they overlap or not. In the file "o.h" there is
// a similar error at the same file offset, but they do not overlap.
