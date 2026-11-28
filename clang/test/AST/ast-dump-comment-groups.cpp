// Without serialization:
// RUN: %clang_cc1 -fretain-comments -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck -strict-whitespace %s
//
// With serialization:
// RUN: %clang_cc1 -fretain-comments -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -fretain-comments -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s
//
// As JSON output
// RUN: %clang_cc1 -fretain-comments -ast-dump=json -ast-dump-filter TestJSON %s \
// RUN: | FileCheck --check-prefix=JSON %s

/**
 * \defgroup IntVariables Global integer variables
 * @{
 */

/** an integer variable */
extern int TestDefGroup;

/**@}*/

extern int TestAfterDefGroup;
// CHECK:      VarDecl{{.*}} TestDefGroup 'int' extern
// CHECK-NEXT: |-FullComment
// CHECK:      `-DocCommentGroup 0x{{[^ ]*}} <line:17:4, line:23:4> defgroup Name="IntVariables" Title="Global integer variables"{{$}}
// CHECK:      VarDecl{{.*}} TestAfterDefGroup 'int' extern
// CHECK-NOT:  DocCommentGroup

/// \name Accessors
/// \{
int TestMemberGroup1();
int TestMemberGroup2();
/// \}
// CHECK:      FunctionDecl{{.*}} TestMemberGroup1 'int ()'
// CHECK:      `-DocCommentGroup 0x{{[^ ]*}} <{{.*}}:5, line:[[@LINE-2]]:5> name Name="Accessors"{{$}}
// CHECK:      FunctionDecl{{.*}} TestMemberGroup2 'int ()'
// CHECK-NEXT: `-DocCommentGroup {{.*}} name Name="Accessors"{{$}}

/// \defgroup Outer
/// @{
/// @{
int TestNested;
/// @}
/// @}
// CHECK:      VarDecl{{.*}} TestNested 'int'
// CHECK:      `-DocCommentGroup 0x{{[^ ]*}} <{{.*}}:5, line:[[@LINE-3]]:5> unnamed{{$}}
// CHECK-NEXT:   `-DocCommentGroup 0x{{[^ ]*}} <line:[[@LINE-7]]:5, line:[[@LINE-3]]:5> defgroup Name="Outer"{{$}}

/// @{
struct TestClass {
  int TestMemberOutsideGroup;
};
/// @}
// CHECK:      CXXRecordDecl{{.*}} struct TestClass definition
// CHECK:      {{[|`]}}-DocCommentGroup {{.*}} unnamed{{$}}
// CHECK:      FieldDecl{{.*}} TestMemberOutsideGroup 'int'{{$}}
// CHECK-NOT:  DocCommentGroup
// CHECK:      VarDecl{{.*}} TestJSON 'int'

/// \weakgroup W
/// @{
int TestJSON;
// JSON:      "kind": "DocCommentGroup",
// JSON:      "groupKind": "weakgroup",
// JSON-NEXT: "name": "W",
// JSON-NEXT: "unterminated": true

/// \name Unterminated
/// @{
int TestUnterminated;
// CHECK:      VarDecl{{.*}} TestUnterminated 'int'
// CHECK:      `-DocCommentGroup 0x{{[^ ]*}} <{{.*}}:5> name Name="Unterminated" unterminated{{$}}
// CHECK-NEXT:   `-DocCommentGroup {{.*}} weakgroup Name="W" unterminated{{$}}
