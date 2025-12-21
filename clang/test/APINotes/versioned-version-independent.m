// RUN: rm -rf %t && mkdir -p %t

// Build and check the module file in version-independent mode.
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -fswift-version-independent-apinotes -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' &> %t/VersionedKit_AST_Dump.txt
// RUN: cat %t/VersionedKit_AST_Dump.txt | FileCheck -check-prefix=CHECK-VERSIONED-DUMP %s

#import <VersionedKit/VersionedKit.h>

// CHECK-VERSIONED-DUMP-LABEL: Dumping moveToPointDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "moveTo(a:b:)"

// CHECK-VERSIONED-DUMP-LABEL: Dumping unversionedRenameDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "unversionedRename_HEADER()"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_NOTES()"

// CHECK-VERSIONED-DUMP-LABEL: Dumping TestGenericDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0
// CHECK-VERSIONED-DUMP-NEXT: SwiftImportAsNonGenericAttr {{.+}} <<invalid sloc>>

// CHECK-VERSIONED-DUMP:  Swift3RenamedOnlyDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Name"

// CHECK-VERSIONED-DUMP: Swift3RenamedAlsoDUMP
// CHECK-VERSIONED-DUMP: SwiftNameAttr {{.+}} "Swift4Name"
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Also"

// CHECK-VERSIONED-DUMP: Swift4RenamedDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedAdditionAttr {{.+}} Implicit 4
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift4Name"

