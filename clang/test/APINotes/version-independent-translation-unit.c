// Verify that non-module/PCH compilations (including precompiled preambles)
// apply API notes directly under -fswift-version-independent-apinotes.

// RUN: rm -rf %t && split-file %s %t
// RUN: %clang_cc1 -fapinotes -fswift-version-independent-apinotes -fapinotes-swift-version=4 -I %t %t/tu.c -ast-dump -ast-dump-filter tuRenamed -x c | FileCheck %s
// RUN: %clang_cc1 -fapinotes -fapinotes-swift-version=4 -I %t %t/tu.c -ast-dump -ast-dump-filter tuRenamed -x c | FileCheck %s

// CHECK: Dumping tuRenamed:
// CHECK: SwiftVersionedAdditionAttr {{.*}} Implicit 4.0 IsReplacedByActive 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.*}} "tuUnversioned()"
// CHECK-NEXT: SwiftNameAttr {{.*}} "tuV4()"
// CHECK-NOT: SwiftVersionedSliceAttr

// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 1 local %t/preamble.c -fapinotes -Xclang -fswift-version-independent-apinotes -Xclang -fapinotes-swift-version=5 -I %t 2>&1 | FileCheck --check-prefix=PREAMBLE %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 1 local %t/preamble.c -fapinotes -Xclang -fapinotes-swift-version=5 -I %t 2>&1 | FileCheck --check-prefix=PREAMBLE %s
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
