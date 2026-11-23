// RUN: rm -rf %t && mkdir -p %t

// Build and check the module file in version-independent mode.
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' &> %t/VersionedKit_AST_Dump.txt
// RUN: cat %t/VersionedKit_AST_Dump.txt | FileCheck -check-prefix=CHECK-VERSIONED-DUMP %s

#import <VersionedKit/VersionedKit.h>

// Each slice an API notes lookup supplied is recorded by a
// SwiftVersionedSliceAttr, whether or not it went on to set a key, and both it
// and the wrappers carry the slice group they belong to. The trailing 0 on
// every line below is that group: one lookup here, because VersionedKit has a
// single reader.

// CHECK-VERSIONED-DUMP-LABEL: Dumping moveToPointDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "moveTo(a:b:)"

// CHECK-VERSIONED-DUMP-LABEL: Dumping unversionedRenameDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "unversionedRename_HEADER()"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_NOTES()"

// The case this attribute exists for: the 3.0 slice names the declaration and
// sets no key. Selecting it suppresses the unversioned rename, and no addition
// wrapper records that, so the bare slice marker is the only evidence the slice
// exists. The -NEXT chain is what makes this a real assertion: it pins the 3.0
// marker directly after the unversioned rename, so no addition wrapper for 3.0
// can sit between them.
// CHECK-VERSIONED-DUMP-LABEL: Dumping keylessSliceDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedSliceAttr {{.+}} Implicit 0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "keylessSlice_NOTES()"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}

// CHECK-VERSIONED-DUMP-LABEL: Dumping TestGenericDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftImportAsNonGenericAttr {{.+}} <<invalid sloc>>

// CHECK-VERSIONED-DUMP:  Swift3RenamedOnlyDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Name"

// CHECK-VERSIONED-DUMP: Swift3RenamedAlsoDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "Swift4Name"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedSliceAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Also"

// CHECK-VERSIONED-DUMP: Swift4RenamedDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedSliceAttr {{.+}} Implicit 4 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 4 0{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift4Name"

