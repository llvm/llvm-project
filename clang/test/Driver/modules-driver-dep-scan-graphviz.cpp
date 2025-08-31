// Tests that the module dependency scan and the module dependency graph
// generation are correct.

// RUN: split-file %s %t

// RUN: %clang -std=c++23 -fmodules -fmodules-driver -Rmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   %t/A.cpp %t/A-B.cpp %t/A-C.cpp %t/B.cpp -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefixes=CHECK %s

// CHECK: clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT:  digraph "Module Dependency Graph" {
// CHECK-NEXT:          label="Module Dependency Graph";
// CHECK-NEXT:          node [shape=Mrecord];

// CHECK:               "transitive1 (Clang module)" [ label="{ Type: Clang module | Provides: transitive1 }"];
// CHECK-NEXT:          "transitive2 (Clang module)" [ label="{ Type: Clang module | Provides: transitive2 }"];
// CHECK-NEXT:          "direct1 (Clang module)" [ label="{ Type: Clang module | Provides: direct1 }"];
// CHECK-NEXT:          "direct2 (Clang module)" [ label="{ Type: Clang module | Provides: direct2 }"];
// CHECK-NEXT:          "root (Clang module)" [ label="{ Type: Clang module | Provides: root }"];
// CHECK-NEXT:          "[[PREFIX]]/main.cpp (Non-module source)" [ label="{ Type: Non-module source | Filename: [[PREFIX]]/main.cpp }"];
// CHECK-NEXT:          "A (C++20 module)" [ label="{ Type: C++20 module | Provides: A | Filename: [[PREFIX]]/A.cpp }"];
// CHECK-NEXT:          "A:B (C++20 module)" [ label="{ Type: C++20 module | Provides: A:B | Filename: [[PREFIX]]/A-B.cpp }"];
// CHECK-NEXT:          "A:C (C++20 module)" [ label="{ Type: C++20 module | Provides: A:C | Filename: [[PREFIX]]/A-C.cpp }"];
// CHECK-NEXT:          "B (C++20 module)" [ label="{ Type: C++20 module | Provides: B | Filename: [[PREFIX]]/B.cpp }"];

// CHECK:               "transitive1 (Clang module)" -> "direct1 (Clang module)";
// CHECK-NEXT:          "transitive1 (Clang module)" -> "direct2 (Clang module)";
// CHECK-NEXT:          "transitive2 (Clang module)" -> "direct1 (Clang module)";
// CHECK-NEXT:          "direct1 (Clang module)" -> "root (Clang module)";
// CHECK-NEXT:          "direct1 (Clang module)" -> "A:B (C++20 module)";
// CHECK-NEXT:          "direct2 (Clang module)" -> "root (Clang module)";
// CHECK-NEXT:          "root (Clang module)" -> "[[PREFIX]]/main.cpp (Non-module source)";
// CHECK-NEXT:          "root (Clang module)" -> "B (C++20 module)";
// CHECK-NEXT:          "A (C++20 module)" -> "[[PREFIX]]/main.cpp (Non-module source)";
// CHECK-NEXT:          "A (C++20 module)" -> "B (C++20 module)";
// CHECK-NEXT:          "A:B (C++20 module)" -> "A (C++20 module)";
// CHECK-NEXT:          "A:C (C++20 module)" -> "A (C++20 module)";
// CHECK-NEXT:          "B (C++20 module)" -> "[[PREFIX]]/main.cpp (Non-module source)";
// CHECK-NEXT:  }

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
