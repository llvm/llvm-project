// Checks that -fmodules-driver correctly handles compilations using both
// Standard C++20 modules and Clang modules.
// Importing a Standard C++20 module into Clang module is not supported yet.

// RUN: split-file %s %t
// RUN: rm -rf %t/modules-cache

// RUN: %clang -c -std=c++23 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -fmodules -Rmodule-import \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp %t/A.cppm %t/A-part1.cppm %t/A-part1-impl.cppm 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-REMARKS %s

// The scan itself will also produce [-Rmodule-import] remarks.
// Let's skip past them, we only care about the final -cc1 commands.
// CHECK-REMARKS:       clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-REMARKS-NEXT:  digraph "Module Dependency Graph" {
// CHECK-REMARKS:       }

// CHECK-REMARKS: [[PREFIX]]/A-part1-impl.cppm:2:2: remark: importing module 'root' from
// CHECK-REMARKS: [[PREFIX]]/A-part1.cppm:2:2: remark: importing module 'root' from
// CHECK-REMARKS: [[PREFIX]]/A.cppm:2:2: remark: importing module 'root' from
// CHECK-REMARKS: [[PREFIX]]/A.cppm:4:8: remark: importing module 'A:part1' from
// CHECK-REMARKS: [[PREFIX]]/A.cppm:4:8: remark: importing module 'root' into 'A:part1' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'root' into 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'A:part1' into 'A' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:1: remark: importing module 'root' into 'A:part1' from

// RUN: %clang -std=c++23 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -fmodules -Rmodule-import \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp %t/A.cppm %t/A-part1.cppm %t/A-part1-impl.cppm \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-CC1 %s

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[ROOTPCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "[[PREFIX]]/module.modulemap"
// CHECK-CC1-SAME: "-fmodule-name=root"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "[[PREFIX]]/A-part1-impl.cppm"
// CHECK-CC1-SAME: "-fmodule-file=root=[[ROOTPCM]]"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PART1_IMPL_PCM:[^"]+]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "[[PREFIX]]/A-part1.cppm"
// CHECK-CC1-SAME: "-fmodule-file=root=[[ROOTPCM]]"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PART1_PCM:[^"]+]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "[[PREFIX]]/A.cppm"
// CHECK-CC1-SAME: "-fmodule-file=root=[[ROOTPCM]]"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-output=[[A_PCM:[^"]+]]"
// CHECK-CC1-SAME: "-fmodule-file=A:part1=[[A_PART1_PCM]]"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "[[PREFIX]]/main.cpp"
// CHECK-CC1-SAME: "-fno-implicit-modules"
// CHECK-CC1-SAME: "-fmodule-file=A=[[A_PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=A:part1=[[A_PART1_PCM]]"

//--- main.cpp
import A;

int main() {
  a();
}

//--- A.cppm
module;
#include "root.h"
export module A;
export import :part1;

export int a() {
  return part1() + root();
}

//--- A-part1.cppm
module;
#include "root.h"
export module A:part1;
export int part1();

//--- A-part1-impl.cppm
module;
#include "root.h"
module A:part1_impl;

int part1() {
  return root();
}

//--- module.modulemap
module root { header "root.h" export * }

//--- root.h
inline int root() { return 1; }
