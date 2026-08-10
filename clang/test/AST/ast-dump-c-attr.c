// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux \
// RUN: -Wno-deprecated-declarations -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux \
// RUN: -Wno-deprecated-declarations -emit-pch -o %t %s
// RUN: %clang_cc1 -x c -triple x86_64-pc-linux \
// RUN: -Wno-deprecated-declarations -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

int Test1 [[deprecated]];
// CHECK:      VarDecl{{.*}}Test1
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:13> "" ""

enum [[deprecated("Frobble")]] Test2 {
  Test3 [[deprecated]]
};
// CHECK:      EnumDecl{{.*}}Test2
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:8, col:28> "Frobble" ""
// CHECK-NEXT:   EnumConstantDecl{{.*}}Test3
// CHECK-NEXT:     DeprecatedAttr 0x{{[^ ]*}} <col:11> "" ""

struct [[deprecated]] Test4 {
  [[deprecated("Frobble")]] int Test5, Test6;
  int Test7 [[deprecated]] : 12;
};
// CHECK:      RecordDecl{{.*}}Test4
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:10> "" ""
// CHECK-NEXT:   FieldDecl{{.*}}Test5
// CHECK-NEXT:     DeprecatedAttr 0x{{[^ ]*}} <col:5, col:25> "Frobble" ""
// CHECK-NEXT:   FieldDecl{{.*}}Test6
// CHECK-NEXT:     DeprecatedAttr 0x{{[^ ]*}} <col:5, col:25> "Frobble" ""
// CHECK-NEXT:   FieldDecl{{.*}}Test7
// CHECK-NEXT:     ConstantExpr{{.*}}'int'
// CHECK-NEXT:       value: Int 12
// CHECK-NEXT:         IntegerLiteral{{.*}}'int' 12
// CHECK-NEXT:     DeprecatedAttr 0x{{[^ ]*}} <col:15> "" ""

struct [[deprecated]] Test8;
// CHECK:      RecordDecl{{.*}}Test8
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:10> "" ""

[[deprecated]] void Test9(int Test10 [[deprecated]]);
// CHECK:      FunctionDecl{{.*}}Test9
// CHECK-NEXT:   ParmVarDecl{{.*}}Test10
// CHECK-NEXT:     DeprecatedAttr 0x{{[^ ]*}} <col:40> "" ""
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:3> "" ""

void Test11 [[deprecated]](void);
// CHECK:      FunctionDecl{{.*}}Test11
// CHECK-NEXT:   DeprecatedAttr 0x{{[^ ]*}} <col:15> "" ""
