// run: clang-tidy -checks=-*,llvm-include-order -header-filter=.* %s \
// run:   -- -isystem %S/Inputs/Headers -I %S/Inputs/overlapping | \
// run:   not grep "note: this fix will not be applied because it overlaps with another fix"

#include "b.h"
#include "a.h"

// The comments above are there to match the offset of the #include with the
// offset of the #includes in the .cpp file.
