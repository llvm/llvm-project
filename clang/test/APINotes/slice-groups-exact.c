// Two lookups against a single API notes reader, which is the case that makes
// the slice group distinct from a reader index.
//
// Sema runs a broad lookup for a global function and, when the sidecar carries
// a `Where: Parameters:` entry, a second exact lookup beside it. Each call runs
// its own version selection and each winner is applied, so the two are separate
// competitions even though they read the same file. If both stamped the same
// group, a consumer that recomputes the selection would pool all four slices
// into one competition, pick a single winner, and silently drop either the
// broad annotation or the exact one.
//
// slice-groups.c covers the other way a second group arises, two readers via
// export_as. This one cannot be expressed with one lookup per reader, so it is
// the test that pins the group to the lookup rather than to the reader.

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers %s -ast-dump -ast-dump-filter sliceGroupExactProbe -x c | FileCheck %s

#include "SliceGroupsExact.h"

// CHECK: Dumping sliceGroupExactProbe:
// CHECK: FunctionDecl {{.+}} imported in SliceGroupsExact sliceGroupExactProbe

// The broad lookup's two slices share one group.
// CHECK: SwiftVersionedSliceAttr {{.+}} Implicit 0 0{{$}}
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "broadUnversioned(_:)"
// CHECK-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "broadV3(_:)"

// The exact lookup's two slices share a different group. The trailing 1 is the
// assertion: it must not be 0, or the two competitions have been pooled.
// CHECK-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 0 1{{$}}
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 1{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "exactUnversioned(_:)"
// CHECK-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 1{{$}}
// CHECK-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 1{{$}}
// CHECK-NEXT: SwiftNameAttr {{.+}} "exactV3(_:)"
