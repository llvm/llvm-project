// Run lines are sensitive to line numbers and come below the code.

#ifndef HEADER
#define HEADER

/// Comment for 'functionBeforeImports'.
void functionBeforeImports(void);

#import <DocCommentsA/DocCommentsA.h>
#import <DocCommentsB/DocCommentsB.h>

@class NSString;

//===---
// rdar://14258334
// Check that we attach comments to properties correctly.
//===---

@interface MyClass {
}

/// property1_isdoxy1 IS_DOXYGEN_SINGLE
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy1;
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy2; ///< property1_isdoxy2 IS_DOXYGEN_SINGLE
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy3; /**< property1_isdoxy3 IS_DOXYGEN_SINGLE */
@property (nonatomic, copy, readwrite) NSString *property1_isdoxy4; /*!< property1_isdoxy4 IS_DOXYGEN_SINGLE */

/// method1_isdoxy1 IS_DOXYGEN_SINGLE
- (void)method1_isdoxy1;
- (void)method1_isdoxy2; ///< method1_isdoxy2 IS_DOXYGEN_SINGLE
- (void)method1_isdoxy3; /**< method1_isdoxy3 IS_DOXYGEN_SINGLE */
- (void)method1_isdoxy4; /*!< method1_isdoxy4 IS_DOXYGEN_SINGLE */
@end

//===---
// rdar://14348912
// Check that we attach comments to enums declared using the NS_ENUM macro.
//===---

#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type

/// An_NS_ENUM_isdoxy1 IS_DOXYGEN_SINGLE
typedef NS_ENUM(int, An_NS_ENUM_isdoxy1) { Red, Green, Blue };

// In the implementation of attaching comments to enums declared using the
// NS_ENUM macro, it is tempting to use the fact that enum decl is embedded in
// the typedef.  Make sure that the heuristic is strong enough that it does not
// attach unrelated comments in the following cases where tag decls are
// embedded in declarators.

#define DECLARE_FUNCTIONS(suffix) \
    /** functionFromMacro IS_DOXYGEN_SINGLE */ \
    void functionFromMacro(void) { \
      typedef struct Struct_notdoxy Struct_notdoxy; \
    } \
    /** functionFromMacroWithSuffix IS_DOXYGEN_SINGLE */ \
    void functionFromMacro##suffix(void) { \
      typedef struct Struct_notdoxy Struct_notdoxy; \
    }

/// IS_DOXYGEN_NOT_ATTACHED
DECLARE_FUNCTIONS(WithSuffix)

/// typedef_isdoxy1 IS_DOXYGEN_SINGLE
typedef struct Struct_notdoxy *typedef_isdoxy1;

#define DECLARE_ENUMS(name) \
  /** enumFromMacro IS_DOXYGEN_SINGLE */ \
  enum enumFromMacro { A }; \
  /** namedEnumFromMacro IS_DOXYGEN_SINGLE */ \
  enum name { B };

/// IS_DOXYGEN_NOT_ATTACHED
DECLARE_ENUMS(namedEnumFromMacro)

#endif

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/module-cache

// Check that we serialize comment source locations properly.
// RUN: %clang_cc1 -emit-pch -o %t/out.pch -F %S/Inputs/Frameworks %s
// RUN: %clang_cc1 -include-pch %t/out.pch -F %S/Inputs/Frameworks -fsyntax-only %s

// RUN: c-index-test -write-pch %t/out.pch -F %S/Inputs/Frameworks %s
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -F %S/Inputs/Frameworks > %t/out.c-index-direct
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -F %S/Inputs/Frameworks -fmodules -fmodules-cache-path=%t/module-cache > %t/out.c-index-modules
// RUN: c-index-test -test-load-tu %t/out.pch all -F %S/Inputs/Frameworks > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-modules
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Declarations without Doxygen comments should not pick up some Doxygen comments.
// WRONG-NOT: notdoxy{{.*}}Comment=
// WRONG-NOT: test{{.*}}Comment=

// Non-Doxygen comments should not be attached to anything.
// WRONG-NOT: NOT_DOXYGEN

// Some Doxygen comments are not attached to anything.
// WRONG-NOT: IS_DOXYGEN_NOT_ATTACHED

// Ensure we don't pick up extra comments.
// WRONG-NOT: IS_DOXYGEN_START{{.*}}IS_DOXYGEN_START{{.*}}BriefComment=
// WRONG-NOT: IS_DOXYGEN_END{{.*}}IS_DOXYGEN_END{{.*}}BriefComment=
//
// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-modules
// RUN: FileCheck %s < %t/out.c-index-pch

// These CHECK lines are not located near the code on purpose.  This test
// checks that documentation comments are attached to declarations correctly.
// Adding a non-documentation comment with CHECK line between every two
// documentation comments will only test a single code path.
//
// CHECK-DAG: annotate-comments-objc.m:7:6: FunctionDecl=functionBeforeImports:{{.*}} BriefComment=[Comment for 'functionBeforeImports'.]
// CHECK-DAG: DocCommentsA.h:2:6: FunctionDecl=functionFromDocCommentsA1:{{.*}} BriefComment=[Comment for 'functionFromDocCommentsA1'.]
// CHECK-DAG: DocCommentsA.h:7:6: FunctionDecl=functionFromDocCommentsA2:{{.*}} BriefComment=[Comment for 'functionFromDocCommentsA2'.]
// CHECK-DAG: DocCommentsB.h:2:6: FunctionDecl=functionFromDocCommentsB1:{{.*}} BriefComment=[Comment for 'functionFromDocCommentsB1'.]
// CHECK-DAG: DocCommentsB.h:7:6: FunctionDecl=functionFromDocCommentsB2:{{.*}} BriefComment=[Comment for 'functionFromDocCommentsB2'.]
// CHECK-DAG: DocCommentsC.h:2:6: FunctionDecl=functionFromDocCommentsC:{{.*}} BriefComment=[Comment for 'functionFromDocCommentsC'.]
// CHECK: annotate-comments-objc.m:23:50: ObjCPropertyDecl=property1_isdoxy1:{{.*}} property1_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:24:50: ObjCPropertyDecl=property1_isdoxy2:{{.*}} property1_isdoxy2 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:25:50: ObjCPropertyDecl=property1_isdoxy3:{{.*}} property1_isdoxy3 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:26:50: ObjCPropertyDecl=property1_isdoxy4:{{.*}} property1_isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:29:9: ObjCInstanceMethodDecl=method1_isdoxy1:{{.*}} method1_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:30:9: ObjCInstanceMethodDecl=method1_isdoxy2:{{.*}} method1_isdoxy2 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:31:9: ObjCInstanceMethodDecl=method1_isdoxy3:{{.*}} method1_isdoxy3 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:32:9: ObjCInstanceMethodDecl=method1_isdoxy4:{{.*}} method1_isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:43:22: EnumDecl=An_NS_ENUM_isdoxy1:{{.*}} An_NS_ENUM_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:43:22: TypedefDecl=An_NS_ENUM_isdoxy1:{{.*}} An_NS_ENUM_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:43:22: EnumDecl=An_NS_ENUM_isdoxy1:{{.*}} An_NS_ENUM_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:62:1: FunctionDecl=functionFromMacro:{{.*}} BriefComment=[functionFromMacro IS_DOXYGEN_SINGLE]
// CHECK: annotate-comments-objc.m:62:1: FunctionDecl=functionFromMacroWithSuffix:{{.*}} BriefComment=[functionFromMacroWithSuffix IS_DOXYGEN_SINGLE]
// CHECK: annotate-comments-objc.m:65:32: TypedefDecl=typedef_isdoxy1:{{.*}} typedef_isdoxy1 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments-objc.m:74:1: EnumDecl=enumFromMacro:{{.*}} BriefComment=[enumFromMacro IS_DOXYGEN_SINGLE]
// CHECK: annotate-comments-objc.m:74:15: EnumDecl=namedEnumFromMacro:{{.*}} BriefComment=[namedEnumFromMacro IS_DOXYGEN_SINGLE]
