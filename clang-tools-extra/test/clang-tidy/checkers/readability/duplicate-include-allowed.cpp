// RUN: %check_clang_tidy %s readability-duplicate-include %t -- \
// RUN:   -header-filter='' \
// RUN:   -config='{CheckOptions: [{key: readability-duplicate-include.IgnoreHeaders, value: "pack_begin.h;pack_end.h"}]}' \
// RUN:   -- -I %S/Inputs/duplicate-include-allowed
//
// This test lives in test/clang-tidy/checkers/
// Inputs for allowed duplicate includes are in test/clang-tidy/checkers/Inputs/duplicate-include-allowed

#include "pack_begin.h"
#include "pack_begin.h" 
// No warning expected

#include "other.h"
#include "other.h" 
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: duplicate include [readability-duplicate-include]

#include "pack_end.h"
#include "pack_end.h" 
// No warning expected
