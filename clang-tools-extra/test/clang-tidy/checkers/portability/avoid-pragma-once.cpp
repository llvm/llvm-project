// RUN: %check_clang_tidy %s portability-avoid-pragma-once %t \
// RUN:   -- --header-filter='.*' --  -I%S/Inputs/avoid-pragma-once

// #pragma once
#include "lib0.h"
// CHECK-MESSAGES: lib0.h:1:1:  warning: avoid 'pragma once' directive; use include guards instead


// # pragma once
#include "lib1.h"
// CHECK-MESSAGES: lib1.h:1:1:  warning: avoid 'pragma once' directive; use include guards instead

// # pragma   once 
#include "lib2.h"
// CHECK-MESSAGES: lib2.h:1:1:  warning: avoid 'pragma once' directive; use include guards instead
