// Tests that the module dependency scan and the module dependency graph
// generation are correct.
// This test does not make use of any system inputs.

// RUN: split-file %s %t

// RUN: %clang -std=c++23 -nostdlib -fmodules \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/A.cpp %t/A-B.cpp %t/A-C.cpp %t/B.cpp -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t %s

// CHECK:       clang: remark: standard modules manifest file not found; import of standard library modules not supported [-Rmodules-driver]
// CHECK:       clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT:  digraph "Module Dependency Graph" {
//
// CHECK:      "transitive1-[[HASH_TRANSITIVE1:.*]]" [fillcolor=1, label="{ Module type: Clang module | Module name: transitive1 | Hash: [[HASH_TRANSITIVE1]] }"];
// CHECK-NEXT: "transitive2-[[HASH_TRANSITIVE2:.*]]" [fillcolor=1, label="{ Module type: Clang module | Module name: transitive2 | Hash: [[HASH_TRANSITIVE2]] }"];
// CHECK-NEXT: "direct1-[[HASH_DIRECT1:.*]]" [fillcolor=1, label="{ Module type: Clang module | Module name: direct1 | Hash: [[HASH_DIRECT1]] }"];
// CHECK-NEXT: "direct2-[[HASH_DIRECT2:.*]]" [fillcolor=1, label="{ Module type: Clang module | Module name: direct2 | Hash: [[HASH_DIRECT2]] }"];
// CHECK-NEXT: "root-[[HASH_ROOT:.*]]" [fillcolor=1, label="{ Module type: Clang module | Module name: root | Hash: [[HASH_ROOT]] }"];
// CHECK-NEXT:  "[[PREFIX]]/main.cpp-[[TRIPLE:.*]]" [fillcolor=3, label="{ Filename: [[PREFIX]]/main.cpp | Triple: [[TRIPLE]] }"];
// CHECK-NEXT: "A-[[TRIPLE]]" [fillcolor=2, label="{ Filename: [[PREFIX]]/A.cpp | Module type: Named module | Module name: A | Triple: [[TRIPLE]] }"];
// CHECK-NEXT: "A:B-[[TRIPLE]]" [fillcolor=2, label="{ Filename: [[PREFIX]]/A-B.cpp | Module type: Named module | Module name: A:B | Triple: [[TRIPLE]] }"];
// CHECK-NEXT: "A:C-[[TRIPLE]]" [fillcolor=2, label="{ Filename: [[PREFIX]]/A-C.cpp | Module type: Named module | Module name: A:C | Triple: [[TRIPLE]] }"];
// CHECK-NEXT: "B-[[TRIPLE]]" [fillcolor=2, label="{ Filename: [[PREFIX]]/B.cpp | Module type: Named module | Module name: B | Triple: [[TRIPLE]] }"];
//
// CHECK:        "transitive1-[[HASH_TRANSITIVE1]]" -> "direct1-[[HASH_DIRECT1]]";
// CHECK-NEXT:   "transitive1-[[HASH_TRANSITIVE1]]" -> "direct2-[[HASH_DIRECT2]]";
// CHECK-NEXT:   "transitive2-[[HASH_TRANSITIVE2]]" -> "direct1-[[HASH_DIRECT1]]";
// CHECK-NEXT:   "direct1-[[HASH_DIRECT1]]" -> "root-[[HASH_ROOT]]";
// CHECK-NEXT:   "direct1-[[HASH_DIRECT1]]" -> "A:B-[[TRIPLE]]";
// CHECK-NEXT:   "direct2-[[HASH_DIRECT2]]" -> "root-[[HASH_ROOT]]";
// CHECK-NEXT:   "root-[[HASH_ROOT]]" -> "[[PREFIX]]/main.cpp-[[TRIPLE]]";
// CHECK-NEXT:   "root-[[HASH_ROOT]]" -> "B-[[TRIPLE]]";
// CHECK-NEXT:   "A-[[TRIPLE]]" -> "[[PREFIX]]/main.cpp-[[TRIPLE]]";
// CHECK-NEXT:   "A-[[TRIPLE]]" -> "B-[[TRIPLE]]";
// CHECK-NEXT:   "A:B-[[TRIPLE]]" -> "A-[[TRIPLE]]";
// CHECK-NEXT:   "A:C-[[TRIPLE]]" -> "A-[[TRIPLE]]";
// CHECK-NEXT:   "B-[[TRIPLE]]" -> "[[PREFIX]]/main.cpp-[[TRIPLE]]"; 
// CHECK-NEXT: }

//--- module.modulemap
module root { header "root.h" }
module direct1 { header "direct1.h" }
module direct2 { header "direct2.h" }
module transitive1 { header "transitive1.h" }
module transitive2 { header "transitive2.h" }

//--- root.h
#include "direct1.h"
#include "direct2.h"

//--- direct1.h
#include "transitive1.h"
#include "transitive2.h"

//--- direct2.h
#include "transitive1.h"

//--- transitive1.h
// empty

//--- transitive2.h
// empty

//--- A.cpp
export module A;
export import :B;
import :C;

//--- A-B.cpp
module;
#include "direct1.h"
export module A:B;

//--- A-C.cpp
export module A:C;

//--- B.cpp
module;
#include "root.h"
export module B;
import A;

//--- main.cpp
#include "root.h"
import A;
import B;
