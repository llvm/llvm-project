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
// RUN:   | FileCheck -DPREFIX=%/t --check-prefixes=CHECK %s

// CHECK:      clang: remark: standard modules manifest file not found; import of standard library modules not supported [-Rmodules-driver]

// CHECK:      clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT: digraph "Module Dependency Graph" {
// CHECK-NEXT:         label="Module Dependency Graph";
// CHECK-NEXT:         node [colorscheme={{.*}},style=filled,shape=Mrecord];
// CHECK-NEXT:         edge [dir="back"];

// CHECK:              "transitive1:[[HASH_TRANSITIVE1:.*]]" [ fillcolor=1,label="{ Kind: Clang module | Module name: transitive1 | Modulemap file: [[PREFIX]]/module.modulemap | Input origin: User | Hash: [[HASH_TRANSITIVE1]] }"];
// CHECK-NEXT:         "transitive2:[[HASH_TRANSITIVE2:.*]]" [ fillcolor=1,label="{ Kind: Clang module | Module name: transitive2 | Modulemap file: [[PREFIX]]/module.modulemap | Input origin: User | Hash: [[HASH_TRANSITIVE2]] }"];
// CHECK-NEXT:         "direct1:[[HASH_DIRECT1:.*]]" [ fillcolor=1,label="{ Kind: Clang module | Module name: direct1 | Modulemap file: [[PREFIX]]/module.modulemap | Input origin: User | Hash: [[HASH_DIRECT1]] }"];
// CHECK-NEXT:         "direct2:[[HASH_DIRECT2:.*]]" [ fillcolor=1,label="{ Kind: Clang module | Module name: direct2 | Modulemap file: [[PREFIX]]/module.modulemap | Input origin: User | Hash: [[HASH_DIRECT2]] }"];
// CHECK-NEXT:         "root:[[HASH_ROOT:.*]]" [ fillcolor=1,label="{ Kind: Clang module | Module name: root | Modulemap file: [[PREFIX]]/module.modulemap | Input origin: User | Hash: [[HASH_ROOT]] }"];
// CHECK-NEXT:         "[[PREFIX]]/main.cpp" [ fillcolor=3,label="{ Kind: Non-module | Filename: [[PREFIX]]/main.cpp }"];
// CHECK-NEXT:         "A" [ fillcolor=2,label="{ Kind: C++ named module | Module name: A | Filename: [[PREFIX]]/A.cpp | Input origin: User | Hash: {{.*}} }"];
// CHECK-NEXT:         "A:B" [ fillcolor=2,label="{ Kind: C++ named module | Module name: A:B | Filename: [[PREFIX]]/A-B.cpp | Input origin: User | Hash: {{.*}} }"];
// CHECK-NEXT:         "A:C" [ fillcolor=2,label="{ Kind: C++ named module | Module name: A:C | Filename: [[PREFIX]]/A-C.cpp | Input origin: User | Hash: {{.*}} }"];
// CHECK-NEXT:         "B" [ fillcolor=2,label="{ Kind: C++ named module | Module name: B | Filename: [[PREFIX]]/B.cpp | Input origin: User | Hash: {{.*}} }"];

// CHECK:              "direct1:[[HASH_DIRECT1]]" -> "transitive1:[[HASH_TRANSITIVE1]]";
// CHECK-NEXT:         "direct1:[[HASH_DIRECT1]]" -> "transitive2:[[HASH_TRANSITIVE2]]";
// CHECK-NEXT:         "direct2:[[HASH_DIRECT2]]" -> "transitive1:[[HASH_TRANSITIVE1]]";
// CHECK-NEXT:         "root:[[HASH_ROOT]]" -> "direct1:[[HASH_DIRECT1]]";
// CHECK-NEXT:         "root:[[HASH_ROOT]]" -> "direct2:[[HASH_DIRECT2]]";
// CHECK-NEXT:         "[[PREFIX]]/main.cpp" -> "root:[[HASH_ROOT]]";
// CHECK-NEXT:         "[[PREFIX]]/main.cpp" -> "A";
// CHECK-NEXT:         "[[PREFIX]]/main.cpp" -> "B";
// CHECK-NEXT:         "A" -> "A:B";
// CHECK-NEXT:         "A" -> "A:C";
// CHECK-NEXT:         "A:B" -> "direct1:[[HASH_DIRECT1]]";
// CHECK-NEXT:         "B" -> "root:[[HASH_ROOT]]";
// CHECK-NEXT:         "B" -> "A";
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
