// RUN: rm -rf %t && mkdir -p %t

// Build and check the unversioned module file.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Unversioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-UNVERSIONED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Unversioned -fdisable-module-hash -fapinotes-modules  -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' | FileCheck -check-prefix=CHECK-DUMP -check-prefix=CHECK-UNVERSIONED-DUMP %s

// Build and check the versioned module file.
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules  -fapinotes-swift-version=3 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -ast-print %t/ModulesCache/Versioned/VersionedKit.pcm | FileCheck -check-prefix=CHECK-VERSIONED %s
// RUN: %clang_cc1 -fmodules -fblocks -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache/Versioned -fdisable-module-hash -fapinotes-modules  -fapinotes-swift-version=3 -fsyntax-only -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -ast-dump -ast-dump-filter 'DUMP' | FileCheck -check-prefix=CHECK-DUMP -check-prefix=CHECK-VERSIONED-DUMP %s

#import <VersionedKit/VersionedKit.h>

// CHECK-UNVERSIONED: void moveToPointDUMP(double x, double y) __attribute__((swift_name("moveTo(x:y:)")));
// CHECK-VERSIONED: void moveToPointDUMP(double x, double y) __attribute__((swift_name("moveTo(a:b:)")));

// CHECK-DUMP-LABEL: Dumping moveToPointDUMP
// CHECK-VERSIONED-DUMP: SwiftVersionedAttr {{.+}} Implicit 3.0 IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} Implicit "moveTo(a:b:)"
// CHECK-UNVERSIONED-DUMP: SwiftNameAttr {{.+}} "moveTo(x:y:)"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} Implicit "moveTo(a:b:)"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping unversionedRenameDUMP
// CHECK-DUMP: in VersionedKit unversionedRenameDUMP
// CHECK-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 0 IsReplacedByActive{{$}}
// CHECK-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_HEADER()"
// CHECK-DUMP-NEXT: SwiftNameAttr {{.+}} "unversionedRename_NOTES()"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping TestGenericDUMP
// CHECK-VERSIONED-DUMP: SwiftImportAsNonGenericAttr {{.+}} Implicit
// CHECK-UNVERSIONED-DUMP: SwiftVersionedAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftImportAsNonGenericAttr {{.+}} Implicit
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift3RenamedOnlyDUMP
// CHECK-DUMP: in VersionedKit Swift3RenamedOnlyDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedRemovalAttr {{.+}} Implicit 3.0 {{[0-9]+}} IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} Implicit "SpecialSwift3Name"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift3RenamedAlsoDUMP
// CHECK-DUMP: in VersionedKit Swift3RenamedAlsoDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 3.0 IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <line:{{.+}}, col:{{.+}}> "Swift4Name"
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift3Also"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} <line:{{.+}}, col:{{.+}}> "Swift4Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 3.0{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} Implicit "SpecialSwift3Also"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-LABEL: Dumping Swift4RenamedDUMP
// CHECK-DUMP: in VersionedKit Swift4RenamedDUMP
// CHECK-VERSIONED-DUMP-NEXT: SwiftVersionedRemovalAttr {{.+}} Implicit 4 {{[0-9]+}} IsReplacedByActive{{$}}
// CHECK-VERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} "SpecialSwift4Name"
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftVersionedAttr {{.+}} Implicit 4{{$}}
// CHECK-UNVERSIONED-DUMP-NEXT: SwiftNameAttr {{.+}} Implicit "SpecialSwift4Name"
// CHECK-DUMP-NOT: Attr

// CHECK-DUMP-NOT: Dumping

// CHECK-UNVERSIONED: void acceptClosure(void (^block)(void) __attribute__((noescape)));
// CHECK-VERSIONED: void acceptClosure(void (^block)(void));

// CHECK-UNVERSIONED: void privateFunc() __attribute__((swift_private));

// CHECK-UNVERSIONED: typedef double MyDoubleWrapper __attribute__((swift_wrapper("struct")));

// CHECK-UNVERSIONED:      enum MyErrorCode {
// CHECK-UNVERSIONED-NEXT:     MyErrorCodeFailed = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((ns_error_domain(MyErrorDomain)));

// CHECK-UNVERSIONED: __attribute__((swift_bridge("MyValueType")))
// CHECK-UNVERSIONED: @interface MyReferenceType

// CHECK-VERSIONED: void privateFunc();

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

// CHECK-UNVERSIONED-LABEL: enum FlagEnum {
// CHECK-UNVERSIONED-NEXT:     FlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     FlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: } __attribute__((flag_enum));
// CHECK-UNVERSIONED-LABEL: enum NewlyFlagEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyFlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     NewlyFlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: } __attribute__((flag_enum));
// CHECK-UNVERSIONED-LABEL: enum APINotedFlagEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedFlagEnumA = 1,
// CHECK-UNVERSIONED-NEXT:     APINotedFlagEnumB = 2
// CHECK-UNVERSIONED-NEXT: } __attribute__((flag_enum));
// CHECK-UNVERSIONED-LABEL: enum OpenEnum {
// CHECK-UNVERSIONED-NEXT:     OpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum NewlyOpenEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum NewlyClosedEnum {
// CHECK-UNVERSIONED-NEXT:     NewlyClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));
// CHECK-UNVERSIONED-LABEL: enum ClosedToOpenEnum {
// CHECK-UNVERSIONED-NEXT:     ClosedToOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum OpenToClosedEnum {
// CHECK-UNVERSIONED-NEXT:     OpenToClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));
// CHECK-UNVERSIONED-LABEL: enum APINotedOpenEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedOpenEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum APINotedClosedEnum {
// CHECK-UNVERSIONED-NEXT:     APINotedClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));

// CHECK-VERSIONED-LABEL: enum FlagEnum {
// CHECK-VERSIONED-NEXT:     FlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     FlagEnumB = 2
// CHECK-VERSIONED-NEXT: } __attribute__((flag_enum));
// CHECK-VERSIONED-LABEL: enum NewlyFlagEnum {
// CHECK-VERSIONED-NEXT:     NewlyFlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     NewlyFlagEnumB = 2
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum APINotedFlagEnum {
// CHECK-VERSIONED-NEXT:     APINotedFlagEnumA = 1,
// CHECK-VERSIONED-NEXT:     APINotedFlagEnumB = 2
// CHECK-VERSIONED-NEXT: } __attribute__((flag_enum));
// CHECK-VERSIONED-LABEL: enum OpenEnum {
// CHECK-VERSIONED-NEXT:     OpenEnumA = 1
// CHECK-VERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-VERSIONED-LABEL: enum NewlyOpenEnum {
// CHECK-VERSIONED-NEXT:     NewlyOpenEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum NewlyClosedEnum {
// CHECK-VERSIONED-NEXT:     NewlyClosedEnumA = 1
// CHECK-VERSIONED-NEXT: };
// CHECK-VERSIONED-LABEL: enum ClosedToOpenEnum {
// CHECK-VERSIONED-NEXT:     ClosedToOpenEnumA = 1
// CHECK-VERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));
// CHECK-VERSIONED-LABEL: enum OpenToClosedEnum {
// CHECK-VERSIONED-NEXT:     OpenToClosedEnumA = 1
// CHECK-VERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-VERSIONED-LABEL: enum APINotedOpenEnum {
// CHECK-VERSIONED-NEXT:     APINotedOpenEnumA = 1
// CHECK-VERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-VERSIONED-LABEL: enum APINotedClosedEnum {
// CHECK-VERSIONED-NEXT:     APINotedClosedEnumA = 1
// CHECK-VERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));

// These don't actually have versioned information, so we just check them once.
// CHECK-UNVERSIONED-LABEL: enum SoonToBeCFEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum SoonToBeNSEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open")));
// CHECK-UNVERSIONED-LABEL: enum SoonToBeCFOptions {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFOptionsA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open"))) __attribute__((flag_enum));
// CHECK-UNVERSIONED-LABEL: enum SoonToBeNSOptions {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSOptionsA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("open"))) __attribute__((flag_enum));
// CHECK-UNVERSIONED-LABEL: enum SoonToBeCFClosedEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeCFClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));
// CHECK-UNVERSIONED-LABEL: enum SoonToBeNSClosedEnum {
// CHECK-UNVERSIONED-NEXT:     SoonToBeNSClosedEnumA = 1
// CHECK-UNVERSIONED-NEXT: } __attribute__((enum_extensibility("closed")));
// CHECK-UNVERSIONED-LABEL: enum UndoAllThatHasBeenDoneToMe {
// CHECK-UNVERSIONED-NEXT:     UndoAllThatHasBeenDoneToMeA = 1
// CHECK-UNVERSIONED-NEXT: };
