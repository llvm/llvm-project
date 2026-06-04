// Checks that -fmodules-driver correctly handles compilations using Clang modules.

// RUN: split-file %s %t
// RUN: rm -rf %t/modules-cache

// RUN: %clang -std=c++23 \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   -fmodules -Rmodule-import \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   -fsyntax-only %t/main.cpp 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t --check-prefix=CHECK-REMARKS %s

// RUN: rm -rf %t/modules-cache
// RUN: %clang -std=c++23 \
// RUN:   -fmodules-driver \
// RUN:   -fmodules \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   -fsyntax-only %t/main.cpp \
// RUN:   -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck --check-prefix=CHECK-CC1 %s

// The scan itself will also produce [-Rmodule-import] remarks.
// Let's skip past them, we only care about the final -cc1 commands.
// CHECK-REMARKS:       clang: remark: printing module dependency graph [-Rmodules-driver]
// CHECK-REMARKS-NEXT:  digraph "Module Dependency Graph" {
// CHECK-REMARKS:       }

// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'root' from
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'direct1' into 'root'
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'transitive1' into 'direct1'
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'transitive2' into 'direct1'
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'direct2' into 'root'
// CHECK-REMARKS: [[PREFIX]]/main.cpp:1:2: remark: importing module 'transitive2' into 'direct2'

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[TRANSITIVE2PCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "-fmodule-name=transitive2"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[DIRECT2PCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "-fmodule-file=transitive2=[[TRANSITIVE2PCM]]"
// CHECK-CC1-SAME: "-fmodule-name=direct2"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[TRANSITIVE1PCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "-fmodule-name=transitive1"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[DIRECT1PCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "-fmodule-file=transitive1=[[TRANSITIVE1PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=transitive2=[[TRANSITIVE2PCM]]"
// CHECK-CC1-SAME: "-fmodule-name=direct1"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-o" "[[ROOTPCM:[^"]+]]"
// CHECK-CC1-SAME: "-emit-module"
// CHECK-CC1-SAME: "-fmodule-file=direct1=[[DIRECT1PCM]]"
// CHECK-CC1-SAME: "-fmodule-file=direct2=[[DIRECT2PCM]]"
// CHECK-CC1-SAME: "-fmodule-name=root"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// CHECK-CC1: "-cc1"
// CHECK-CC1-SAME: "-fsyntax-only"
// CHECK-CC1-SAME: "{{.*}}/main.cpp"
// CHECK-CC1-SAME: "-fmodule-file=root=[[ROOTPCM]]"
// CHECK-CC1-SAME: "-fno-implicit-modules"

// (Because of missing include guards, this example would also run into
// redefinition errors when compiling without modules.)

/--- module.modulemap
module root { header "root.h"}
module transitive1 { header "transitive1.h" }
module transitive2 { header "transitive2.h" }
module direct1 { header "direct1.h" }
module direct2 { header "direct2.h" }

//--- root.h
#include "direct1.h"
#include "direct2.h"
int fromRoot() {
  return fromDirect1() + fromDirect2();
}

//--- direct1.h
#include "transitive1.h"
#include "transitive2.h"

int fromDirect1() {
  return fromTransitive1() + fromTransitive2();
}

//--- direct2.h
#include "transitive2.h"

int fromDirect2() {
  return fromTransitive2() + 2;
}

//--- transitive1.h
int fromTransitive1() {
    return 20;
}

//--- transitive2.h
int fromTransitive2() {
    return 10;
}

//--- main.cpp
#include "root.h"

int main() {
 fromRoot();
}
