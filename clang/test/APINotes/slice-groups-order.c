// Two API notes readers and a parameter selector on one declaration, which is
// the only configuration where the slice group numbering's ordering property is
// observable.
//
// Sema makes two lookups per reader: a broad one, and a parameter-selector one
// for a `Where: Parameters:` entry. It applies them reader by reader, broad
// first, so each reader owns an adjacent pair of groups. ExportAsCore takes 0
// and 1, ExportAs takes 2 and 3, and that is also the order Sema applied them.
//
// The ordering matters because a group's winner can collide with another
// group's winner on the same key, as all four do here on SwiftName. Clang
// resolves that by last-applied-wins, so a consumer has to replay the groups in
// ascending order. Numbering the readers 0 and 1 and the selector lookups
// 2 and 3 would apply them in the order 0, 2, 1, 3, and a consumer sorting on
// the ordinal would pick the wrong winner.
//
// slice-groups.c covers two readers with no selector, and slice-groups-exact.c
// covers one reader with a selector. Neither pins the order, because with a
// single pair any numbering is ascending.

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter sliceGroupOrderProbe -x c | FileCheck %s

#include "ExportAs.h"

// CHECK: Dumping sliceGroupOrderProbe:
// CHECK: FunctionDecl {{.+}} imported in ExportAsCore sliceGroupOrderProbe

// Group 0: ExportAsCore's broad lookup, with an unversioned and a 3.0 slice.
// CHECK: SwiftVersionedAdditionAttr {{.+}} Implicit 0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "coreBroad(_:)"
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "coreBroadV3(_:)"

// Group 1: ExportAsCore's parameter-selector lookup. Odd, and adjacent to its
// own reader's broad group rather than pooled with the other reader's.
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 1{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "coreExact(_:)"

// Group 2: ExportAs's broad lookup, reached through export_as.
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 2{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "exportBroad(_:)"

// Group 3: ExportAs's parameter-selector lookup. Applied last, so under the
// legacy selection this is the name that would win.
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 3{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "exportExact(_:)"
