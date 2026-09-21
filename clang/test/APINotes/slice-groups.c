// A declaration annotated by two API notes readers at once, which is what makes
// the slice group observable. ExportAsCore is `export_as ExportAs`, so
// tryAPINotes loads ExportAsCore.apinotes and then ExportAs.apinotes from the
// same directory. Both are public readers.
//
// Clang runs version selection once per lookup and applies every winner, so the
// two readers are two competitions. The group ordinal is what tells a consumer
// which slices are rivals; pooling them would let one reader's slice suppress
// the other's. Every other test in this directory has a single reader, so this
// is the only one where a wrong or swapped group is visible at all.

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter sliceGroupProbe -x c | FileCheck %s

#include "ExportAs.h"

// CHECK: Dumping sliceGroupProbe:
// CHECK: VarDecl {{.+}} imported in ExportAsCore sliceGroupProbe 'int'

// Group 0 is ExportAsCore.apinotes: an unversioned slice and a 3.0 slice.
// CHECK: SwiftVersionedAdditionAttr {{.+}} Implicit 0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "fromCoreUnversioned"
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "fromCoreV3"

// Group 2 is ExportAs.apinotes, reached through export_as. Reader 1's broad
// lookup is group 2, not 1, because each reader reserves an odd number for its
// exact parameter-selector lookup. The nonzero group is the whole point of this
// test: nothing else in the suite produces one, so a bug that collapsed the two
// readers onto one group, or swapped them, would pass everywhere else.
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 2{{$}}
// CHECK-NEXT: SwiftPrivateAttr
// CHECK-NEXT: SwiftVersionedRemovalAttr {{.+}} Implicit 4.0 {{[0-9]+}} 2{{$}}
