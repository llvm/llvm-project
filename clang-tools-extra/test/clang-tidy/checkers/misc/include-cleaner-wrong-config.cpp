// RUN: %check_clang_tidy %s misc-include-cleaner %t \
// RUN: -config='{CheckOptions: \
// RUN:  {"misc-include-cleaner.UnusedIncludes": false,\
// RUN:   "misc-include-cleaner.MissingIncludes": false,\
// RUN:  }}' -- -I%S/Inputs -isystem%S/Inputs/system -fno-delayed-template-parsing

// CHECK-MESSAGES: warning: The check 'misc-include-cleaner' will not perform any analysis because 'UnusedIncludes' and 'MissingIncludes' are both false. [clang-tidy-config]

#include "bar.h"
// CHECK-FIXES-NOT: {{^}}#include "baz.h"{{$}}
