// RUN: %check_clang_tidy %s readability-duplicate-include %t -- \
// RUN:   -config="{CheckOptions: {readability-duplicate-include.IgnoredFilesList: 'pack_.*\\.h'}}" \
// RUN:   -- -I %S/Inputs/duplicate-include

#include "pack_begin.h"
struct A { int x; };
#include "pack_end.h"

// no warning
