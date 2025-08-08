// Tests that the module dependency scan and the module dependency graph
// generation are correct.

// RUN: split-file %s %t

// RUN: %clang -std=c++23 -fmodules -fmodules-driver -Rmodules-driver \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/main.cpp \
// RUN:   %t/A.cpp %t/A-B.cpp %t/A-C.cpp %t/B.cpp > %t/result 2>&1
// RUN: cat %t/result | FileCheck -DPREFIX=%t --check-prefixes=CHECK %s

// CHECK: clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-NEXT: digraph "Module Dependency Graph" {
// CHECK-NEXT:         label="Module Dependency Graph";
// CHECK-NEXT:         node [shape=Mrecord];
// CHECK-NEXT:         edge [dir="back"];
//
// CHECK:              "Clang Module 'transitive1'" [ label = "{ Type: Clang Module | Provides: \"transitive1\" }" ];
// CHECK-NEXT:         "Clang Module 'transitive2'" [ label = "{ Type: Clang Module | Provides: \"transitive2\" }" ];
// CHECK-NEXT:         "Clang Module 'direct1'" [ label = "{ Type: Clang Module | Provides: \"direct1\" }" ];
// CHECK-NEXT:         "Clang Module 'direct2'" [ label = "{ Type: Clang Module | Provides: \"direct2\" }" ];
// CHECK-NEXT:         "Clang Module 'root'" [ label = "{ Type: Clang Module | Provides: \"root\" }" ];
// CHECK-NEXT:         "Non-Module TU '[[PREFIX]]/main.cpp'" [ label = "{ Type: Default TU | Filename: \"[[PREFIX]]/main.cpp\" }" ];
// CHECK-NEXT:         "C++ Named Module 'B'" [ label = "{ Type: C++ Named Module | Filename: \"[[PREFIX]]/B.cpp\" | Provides: \"B\" }" ];
// CHECK-NEXT:         "C++ Named Module 'A'" [ label = "{ Type: C++ Named Module | Filename: \"[[PREFIX]]/A.cpp\" | Provides: \"A\" }" ];
// CHECK-NEXT:         "C++ Named Module 'A:B'" [ label = "{ Type: C++ Named Module | Filename: \"[[PREFIX]]/A-B.cpp\" | Provides: \"A:B\" }" ];
// CHECK-NEXT:         "C++ Named Module 'A:C'" [ label = "{ Type: C++ Named Module | Filename: \"[[PREFIX]]/A-C.cpp\" | Provides: \"A:C\" }" ];
//
// CHECK:              "Clang Module 'direct1'" -> "Clang Module 'transitive1'";
// CHECK-NEXT:         "Clang Module 'direct1'" -> "Clang Module 'transitive2'";
// CHECK-NEXT:         "Clang Module 'direct2'" -> "Clang Module 'transitive1'";
// CHECK-NEXT:         "Clang Module 'root'" -> "Clang Module 'direct1'";
// CHECK-NEXT:         "Clang Module 'root'" -> "Clang Module 'direct2'";
// CHECK-NEXT:         "Non-Module TU '[[PREFIX]]/main.cpp'" -> "C++ Named Module 'B'";
// CHECK-NEXT:         "Non-Module TU '[[PREFIX]]/main.cpp'" -> "Clang Module 'root'";
// CHECK-NEXT:         "C++ Named Module 'B'" -> "C++ Named Module 'A'";
// CHECK-NEXT:         "C++ Named Module 'B'" -> "Clang Module 'root'";
// CHECK-NEXT:         "C++ Named Module 'A'" -> "C++ Named Module 'A:B'";
// CHECK-NEXT:         "C++ Named Module 'A'" -> "C++ Named Module 'A:C'";
// CHECK-NEXT:         "C++ Named Module 'A:B'" -> "Clang Module 'direct1'";
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
import B;

