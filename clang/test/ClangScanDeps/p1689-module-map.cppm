// UNSUPPORTED: system-windows

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -format=p1689 -- \
// RUN:   %clang++ -std=c++20 -c %t/include-only.cpp -o %t/include-only.o \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=INCLUDE-ONLY
// RUN: clang-scan-deps --mode=preprocess-dependency-directives -format=p1689 -- \
// RUN:   %clang++ -std=c++20 -c %t/include-only.cpp -o %t/include-only.o \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=INCLUDE-ONLY

// RUN: clang-scan-deps -format=p1689 -- \
// RUN:   %clang++ -std=c++20 -c %t/mixed.cpp -o %t/mixed.o \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=MIXED

// RUN: sed "s|DIR|%/t|g" %t/compile_commands.json.in > %t/compile_commands.json
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -compilation-database %t/compile_commands.json \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=INCLUDE-ONLY

// Check that this does not enable single-module parse mode, which would skip
// all branches of a conditional whose controlling macro is undefined.
// RUN: clang-scan-deps -format=p1689 -- \
// RUN:   %clang++ -std=c++20 -c %t/conditional.cpp -o %t/conditional.o \
// RUN:   -MD -MT %t/conditional.o -MF %t/conditional.dep > /dev/null
// RUN: cat %t/conditional.dep \
// RUN:   | FileCheck %s -DPREFIX=%/t --check-prefix=CONDITIONAL

// CONDITIONAL-NOT: false.h
// CONDITIONAL:     [[PREFIX]]/true.h

// INCLUDE-ONLY:      {
// INCLUDE-ONLY-NEXT:   "revision": 0,
// INCLUDE-ONLY-NEXT:   "rules": [
// INCLUDE-ONLY-NEXT:     {
// INCLUDE-ONLY-NEXT:       "primary-output": "[[PREFIX]]/include-only.o",
// INCLUDE-ONLY-NEXT:       "requires": [
// INCLUDE-ONLY-NEXT:         {
// INCLUDE-ONLY-NEXT:           "logical-name": "alpha"
// INCLUDE-ONLY-NEXT:         },
// INCLUDE-ONLY-NEXT:         {
// INCLUDE-ONLY-NEXT:           "logical-name": "beta.gamma"
// INCLUDE-ONLY-NEXT:         }
// INCLUDE-ONLY-NEXT:       ]
// INCLUDE-ONLY-NEXT:     }
// INCLUDE-ONLY-NEXT:   ],
// INCLUDE-ONLY-NEXT:   "version": 1
// INCLUDE-ONLY-NEXT: }

// MIXED:      {
// MIXED-NEXT:   "revision": 0,
// MIXED-NEXT:   "rules": [
// MIXED-NEXT:     {
// MIXED-NEXT:       "primary-output": "[[PREFIX]]/mixed.o",
// MIXED-NEXT:       "requires": [
// MIXED-NEXT:         {
// MIXED-NEXT:           "logical-name": "alpha"
// MIXED-NEXT:         }
// MIXED-NEXT:       ]
// MIXED-NEXT:     }
// MIXED-NEXT:   ],
// MIXED-NEXT:   "version": 1
// MIXED-NEXT: }

//--- module.modulemap
module alpha {
  header "alpha-1.h"
  header "alpha-2.h"
}
module beta {
  module gamma {
    header "beta.h"
  }
}
module unused {
  header "unused.h"
}

//--- alpha-1.h
#pragma once
#error mapped headers must not be textually included

//--- alpha-2.h
#pragma once
#error mapped headers must not be textually included

//--- beta.h
#pragma once
#error mapped headers must not be textually included

//--- unused.h
#pragma once

//--- unmapped.h
#pragma once

//--- false.h
#pragma once

//--- true.h
#pragma once

//--- include-only.cpp
#include "alpha-1.h"
#include "alpha-2.h"
#include "beta.h"
#include "unmapped.h"

//--- mixed.cpp
#include "alpha-1.h"
import alpha;

//--- conditional.cpp
#if UNDEFINED
#include "false.h"
#else
#include "true.h"
#endif

//--- compile_commands.json.in
[
  {
    "directory": "DIR",
    "command": "clang++ -std=c++20 -c DIR/include-only.cpp -o DIR/include-only.o -fmodule-map-file=DIR/module.modulemap",
    "file": "DIR/include-only.cpp",
    "output": "DIR/include-only.o"
  }
]
