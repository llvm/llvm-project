// A translation unit that does not build a module applies API notes normally
// under -fswift-version-independent-apinotes. It has one Swift version, so
// there is no module file to serve several versions and nothing to capture for.
// The same checks run against the default mode.

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fapinotes -fswift-version-independent-apinotes -fapinotes-swift-version=4 -I %t %t/tu.c -ast-dump -ast-dump-filter tuRenamed -x c | FileCheck %s
// RUN: %clang_cc1 -fapinotes -fapinotes-swift-version=4 -I %t %t/tu.c -ast-dump -ast-dump-filter tuRenamed -x c | FileCheck %s

// The 4.0 slice is applied and the unversioned one superseded. No slice
// markers, which only a capturing module build emits.
// CHECK: Dumping tuRenamed:
// CHECK: SwiftVersionedAdditionAttr {{.*}} Implicit 4.0 IsReplacedByActive 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.*}} "tuUnversioned()"
// CHECK-NEXT: SwiftNameAttr {{.*}} "tuV4()"
// CHECK-NOT: SwiftVersionedSliceAttr

// A precompiled preamble is the start of its translation unit, so it applies
// API notes too, and its inline body can't use what they make unavailable.
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 1 local %t/preamble.c -fapinotes -Xclang -fswift-version-independent-apinotes -Xclang -fapinotes-swift-version=5 -I %t 2>&1 | FileCheck --check-prefix=PREAMBLE %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 1 local %t/preamble.c -fapinotes -Xclang -fapinotes-swift-version=5 -I %t 2>&1 | FileCheck --check-prefix=PREAMBLE %s
// Once for the first parse, and once for the reparse, which uses the preamble.
// PREAMBLE: Preamble.h:2:{{[0-9]+}}: error: 'preambleGone' is unavailable: gone
// PREAMBLE: Preamble.h:2:{{[0-9]+}}: error: 'preambleGone' is unavailable: gone

//--- APINotes.apinotes
Name: TranslationUnit
Functions:
  - Name: tuRenamed
    SwiftName: 'tuUnversioned()'
  - Name: preambleGone
    Availability: none
    AvailabilityMsg: 'gone'
SwiftVersions:
  - Version: 4.0
    Functions:
      - Name: tuRenamed
        SwiftName: 'tuV4()'

//--- TranslationUnit.h
void tuRenamed(void);

//--- tu.c
#include "TranslationUnit.h"

//--- Preamble.h
void preambleGone(void);
static inline void preambleUser(void) { preambleGone(); }

//--- preamble.c
#include "Preamble.h"
int main(void) { return 0; }
