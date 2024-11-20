// RUN: rm -rf %t && mkdir -p %t

// Build and check the unversioned module file.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Unversioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-UNVERSIONED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' | FileCheck -check-prefix=CHECK-DUMP -check-prefix=CHECK-UNVERSIONED-DUMP %s

// Build and check the versioned module file.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -fapinotes-swift-version=3 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules -fapinotes-swift-version=3 -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' | FileCheck -check-prefix=CHECK-DUMP -check-prefix=CHECK-VERSIONED-DUMP %s

#import <VersionedKit/VersionedKit.h>

// CHECK-UNVERSIONED: void moveToPointDUMP(double x, double y) __attribute__((swift_name("moveTo(x:y:)")));
// CHECK-VERSIONED:__attribute__((swift_name("moveTo(a:b:)"))) void moveToPointDUMP(double x, double y);

// CHECK-DUMP-LABEL: Dumping moveToPointDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "moveTo(a:b:)"
// CHECK-UNVERSIONED-DUMP: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "moveTo(a:b:)"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping unversionedRenameDUMP
// CHECK-DUMP: in VersionedKit unversionedRenameDUMP
// CHECK-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 0 IsReplacedByActive{{$}}
// CHECK-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_HEADER()"
// CHECK-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_NOTES()"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping TestGenericDUMP
// CHECK-VERSIONED-DUMP: SwiftImportAsNonGenericAttr {{.+}} <<invalid sloc>>
// CHECK-UNVERSIONED-DUMP: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftImportAsNonGenericAttr {{.+}} <<invalid sloc>>
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift3RenamedOnlyDUMP
// CHECK-DUMP: in VersionedKit Swift3RenamedOnlyDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedRemovalAttr {{.+}} Implicit 3.0 {{[0-9]+}} IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SpecialSwift3Name"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift3RenamedAlsoDUMP
// CHECK-DUMP: in VersionedKit Swift3RenamedAlsoDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0 IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <line:{{.+}}, col:{{.+}}> "Swift4Name"
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Also"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <line:{{.+}}, col:{{.+}}> "Swift4Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SpecialSwift3Also"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift4RenamedDUMP
// CHECK-DUMP: in VersionedKit Swift4RenamedDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedRemovalAttr {{.+}} Implicit 4 {{[0-9]+}} IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift4Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAdditionAttr {{.+}} Implicit 4{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <<invalid sloc>> "SpecialSwift4Name"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-NOT: Dumping

// CHECK-UNVERSIONED: void acceptClosure(__attribute__((noescape)) void (^block)(void));
// CHECK-VERSIONED: void acceptClosure(void (^block)(void));

// CHECK-UNVERSIONED: void privateFunc(void) __attribute__((swift_private));

// CHECK-UNVERSIONED: typedef double MyDoubleWrapper __attribute__((swift_wrapper("struct")));

// CHECK-UNVERSIONED:      enum __attribute__((ns_error_domain(MyErrorDomain))) MyErrorCode {
// CHECK-UNVERSIONED-NEXT:     MyErrorCodeFailed = 1
// CHECK-UNVERSIONED-NEXT: };

// CHECK-UNVERSIONED: __attribute__((swift_bridge("MyValueType")))
// CHECK-UNVERSIONED: @interface MyReferenceType

// CHECK-VERSIONED: void privateFunc(void);

// CHECK-VERSIONED: typedef double MyDoubleWrapper;

// CHECK-VERSIONED:      enum MyErrorCode {
// CHECK-VERSIONED-NEXT:     MyErrorCodeFailed = 1
// CHECK-VERSIONED-NEXT: };

// CHECK-VERSIONED-NOT: __attribute__((swift_bridge("MyValueType")))
// CHECK-VERSIONED: @interface MyReferenceType

// CHECK-UNVERSIONED: __attribute__((swift_objc_members)
// CHECK-UNVERSIONED-NEXT: @interface TestProperties
// CHECK-VERSIONED-NOT: __attribute__((swift_objc_members)
// CHECK-VERSIONED: @interface TestProperties

// CHECK-UNVERSIONED-LABEL: enum __attribute__((flag_enum)) FlagEnum {
// CHECK-UNVERSIONED-NEXT:     FlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     FlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum __attribute__((flag_enum)) NewlyFlagEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyFlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     NewlyFlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum __attribute__((flag_enum)) APINotedFlagEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedFlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     APINotedFlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) OpenEnum {
// CHECK-UNVERSIONED-NEXT:     OpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) NewlyOpenEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("closed"))) NewlyClosedEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) ClosedToOpenEnum {
// CHECK-UNVERSIONED-NEXT:     ClosedToOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("closed"))) OpenToClosedEnum {
// CHECK-UNVERSIONED-NEXT:     OpenToClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) APINotedOpenEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("closed"))) APINotedClosedEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: };

// CHECK-VERSIONED-LABEL: enum __attribute__((flag_enum)) FlagEnum {
// CHECK-VERSIONED-NEXT:     FlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     FlagEnumB = 2
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum NewlyFlagEnum {
// CHECK-VERSIONED-NEXT:     NewlyFlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     NewlyFlagEnumB = 2
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((flag_enum)) APINotedFlagEnum {
// CHECK-VERSIONED-NEXT:     APINotedFlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     APINotedFlagEnumB = 2
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((enum_extensibility("open"))) OpenEnum {
// CHECK-VERSIONED-NEXT:     OpenEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum NewlyOpenEnum {
// CHECK-VERSIONED-NEXT:     NewlyOpenEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum NewlyClosedEnum {
// CHECK-VERSIONED-NEXT:     NewlyClosedEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((enum_extensibility("closed"))) ClosedToOpenEnum {
// CHECK-VERSIONED-NEXT:     ClosedToOpenEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((enum_extensibility("open"))) OpenToClosedEnum {
// CHECK-VERSIONED-NEXT:     OpenToClosedEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((enum_extensibility("open"))) APINotedOpenEnum {
// CHECK-VERSIONED-NEXT:     APINotedOpenEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum __attribute__((enum_extensibility("closed"))) APINotedClosedEnum {
// CHECK-VERSIONED-NEXT:     APINotedClosedEnumA = 1
// CHECK-VERSIONED-NEXT: };

// These don't actually have versioned information, so we just check them once.
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) SoonToBeCFEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) SoonToBeNSEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) __attribute__((flag_enum)) SoonToBeCFOptions {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFOptionsA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("open"))) __attribute__((flag_enum)) SoonToBeNSOptions {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSOptionsA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("closed"))) SoonToBeCFClosedEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum  __attribute__((enum_extensibility("closed"))) SoonToBeNSClosedEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: };
// CHECK-UNVERSIONED-LABEL: enum UndoAllThatHasBeenDoneToMe {
// CHECK-UNVERSIONED-NEXT:     UndoAllThatHasBeenDoneToMeA = 1
// CHECK-UNVERSIONED-NEXT: };
