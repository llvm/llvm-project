// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name TwoSubs > %t/TwoSubs.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name WithExplicit > %t/WithExplicit.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu1.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 > %t/tu2.rsp

// Extract include-tree casids
// RUN: cat %t/TwoSubs.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/TwoSubs.casid
// RUN: cat %t/WithExplicit.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/WithExplicit.casid
// RUN: cat %t/tu1.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu1.casid
// RUN: cat %t/tu2.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu2.casid

// RUN: echo "MODULE TwoSubs" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/TwoSubs.casid >> %t/result.txt
// RUN: echo "MODULE WithExplicit" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/WithExplicit.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT 1" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu1.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT 2" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu2.casid >> %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// Build the include-tree commands
// RUN: %clang @%t/TwoSubs.rsp
// RUN: %clang @%t/WithExplicit.rsp
// RUN: not %clang @%t/tu1.rsp 2>&1 | FileCheck %s -check-prefix=TU1
// RUN: not %clang @%t/tu2.rsp 2>&1 | FileCheck %s -check-prefix=TU2

// CHECK: MODULE TwoSubs
// CHECK: 2:1 [[PREFIX]]/Sub1.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: TwoSubs.Sub1
// CHECK: 3:1 [[PREFIX]]/Sub2.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: TwoSubs.Sub2

// CHECK: MODULE WithExplicit
// CHECK: 2:1 [[PREFIX]]/TopLevel.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: WithExplicit
// CHECK: 3:1 [[PREFIX]]/Implicit.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: WithExplicit.Implicit
// CHECK: 4:1 [[PREFIX]]/Explicit.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: WithExplicit.Explicit

// CHECK: TRANSLATION UNIT 1
// CHECK: [[PREFIX]]/tu1.c llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 (Module) TwoSubs.Sub1
// CHECK: 3:1 (Module) WithExplicit

// CHECK: TRANSLATION UNIT 2
// CHECK: [[PREFIX]]/tu2.c llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 (Module) TwoSubs.Sub2
// CHECK: 3:1 (Module) WithExplicit.Explicit

//--- cdb.json.template
[
{
  "file": "DIR/tu1.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu1.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
{
  "file": "DIR/tu2.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu2.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
]

//--- module.modulemap
module TwoSubs {
  module Sub1 { header "Sub1.h" }
  module Sub2 { header "Sub2.h" }
}

module WithExplicit {
  header "TopLevel.h"
  module Implicit { header "Implicit.h" }
  explicit module Explicit { header "Explicit.h" }
}

//--- Sub1.h
void sub1(void);

//--- Sub2.h
void sub2(void);

//--- TopLevel.h
void top(void);

//--- Implicit.h
void implicit(void);

//--- Explicit.h
void explicit(void);

//--- tu1.c
#include "Sub1.h"
#include "TopLevel.h"

void tu1(void) {
  top();
  sub1();
  implicit();
  sub2();
// TU1-NOT: error:
// TU1: error: call to undeclared function 'sub2'
// TU1: error: missing '#include "Sub2.h"'
// TU1: note: declaration here is not visible
  explicit();
// TU1: error: call to undeclared function 'explicit'
// TU1: error: missing '#include "Explicit.h"'
// TU1: Explicit.h:1:6: note: declaration here is not visible
}

//--- tu2.c
#include "Sub2.h"
#include "Explicit.h"

void tu2(void) {
  sub2();
  explicit();
// TU2-NOT: error:
  top();
// TU2: error: call to undeclared function 'top'
// TU2: error: missing '#include "TopLevel.h"'
// TU2: note: declaration here is not visible
  sub1();
// TU2: error: call to undeclared function 'sub1'
// TU2: error: missing '#include "Sub1.h"'
// TU2: note: declaration here is not visible
  implicit();
// TU2: error: call to undeclared function 'implicit'
// TU2: error: missing '#include "Implicit.h"'
// TU2: note: declaration here is not visible
}
