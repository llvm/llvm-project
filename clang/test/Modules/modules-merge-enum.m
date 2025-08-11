// RUN: rm -rf %t
// RUN: split-file %s %t


// Expect no crash
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/modcache -fmodule-map-file=%t/module.modulemap %t/source.m

// Add -ast-dump-all to check that the AST nodes are merged correctly.
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/modcache -fmodule-map-file=%t/module.modulemap %t/source.m -ast-dump-all 2>&1 | FileCheck %s


//--- shared.h
// This header is shared between two modules, but it's not a module itself.
// The enums defined here are parsed in both modules, and merged while building ModB.

typedef enum MyEnum1 { MyVal_A } MyEnum1;
// CHECK:      |-EnumDecl 0x{{.*}} imported in ModA.ModAFile1 <undeserialized declarations> MyEnum1
// CHECK-NEXT: | |-also in ModB
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} imported in ModA.ModAFile1 referenced MyVal_A 'int'
// CHECK-NEXT: |-TypedefDecl 0x{{.*}} imported in ModA.ModAFile1 hidden MyEnum1 'enum MyEnum1'
// CHECK-NEXT: | `-EnumType 0x{{.*}} 'enum MyEnum1' imported
// CHECK-NEXT: |   `-Enum 0x{{.*}} 'MyEnum1'


enum MyEnum2 { MyVal_B };
// CHECK:      |-EnumDecl 0x{{.*}} imported in ModA.ModAFile1 <undeserialized declarations> MyEnum2
// CHECK-NEXT: | |-also in ModB
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} imported in ModA.ModAFile1 referenced MyVal_B 'int'


typedef enum { MyVal_C } MyEnum3;
// CHECK:      |-EnumDecl 0x{{.*}} imported in ModA.ModAFile1 <undeserialized declarations>
// CHECK-NEXT: | |-also in ModB
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} imported in ModA.ModAFile1 referenced MyVal_C 'int'
// CHECK-NEXT: |-TypedefDecl 0x{{.*}} imported in ModA.ModAFile1 hidden MyEnum3 'enum MyEnum3'
// CHECK-NEXT: | `-EnumType 0x{{.*}} 'enum MyEnum3' imported
// CHECK-NEXT: |   `-Enum 0x{{.*}}

struct MyStruct {
  enum MyEnum5 { MyVal_D } Field;
};

// CHECK:      |-RecordDecl 0x{{.*}} imported in ModA.ModAFile1 <undeserialized declarations> struct MyStruct definition
// CHECK-NEXT: | |-also in ModB
// CHECK-NEXT: | |-EnumDecl 0x{{.*}} imported in ModA.ModAFile1 <undeserialized declarations> MyEnum5
// CHECK-NEXT: | | |-also in ModB
// CHECK-NEXT: | | `-EnumConstantDecl 0x{{.*}} imported in ModA.ModAFile1 referenced MyVal_D 'int'
// CHECK-NEXT: | `-FieldDecl 0x{{.*}} imported in ModA.ModAFile1 hidden Field 'enum MyEnum5'

// In this case, no merging happens on the EnumDecl in Objective-C, and ASTWriter writes both EnumConstantDecls when building ModB.
enum { MyVal_E };
// CHECK:      |-EnumDecl 0x{{.*}} imported in ModA.ModAFile1 hidden <undeserialized declarations>
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} imported in ModA.ModAFile1 hidden MyVal_E 'int'


// Redeclarations coming from ModB.
// CHECK:      |-TypedefDecl 0x{{.*}} prev 0x{{.*}} imported in ModB MyEnum1 'enum MyEnum1'
// CHECK-NEXT: | `-EnumType 0x{{.*}} 'enum MyEnum1' imported
// CHECK-NEXT: |   `-Enum 0x{{.*}} 'MyEnum1'

// CHECK:      |-EnumDecl 0x{{.*}} prev 0x{{.*}} imported in ModB <undeserialized declarations>
// CHECK-NEXT: | |-also in ModB
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} imported in ModB MyVal_C 'int'
// CHECK-NEXT: |-TypedefDecl 0x{{.*}} prev 0x{{.*}} imported in ModB MyEnum3 'enum MyEnum3'
// CHECK-NEXT: | `-EnumType 0x{{.*}} 'enum MyEnum3' imported
// CHECK-NEXT: |   `-Enum 0x{{.*}}

// CHECK:      |-EnumDecl 0x{{.*}} imported in ModB <undeserialized declarations>
// CHECK-NEXT: | `-EnumConstantDecl 0x{{.*}} first 0x{{.*}} imported in ModB referenced MyVal_E 'int'



//--- module.modulemap
module ModA {
  module ModAFile1 {
    header "ModAFile1.h"
    export *
  }
  module ModAFile2 {
    header "ModAFile2.h"
    export *
  }
}
// The goal of writing ModB is to test that ASTWriter can handle the merged AST nodes correctly.
// For example, a stale declaration in IdResolver can cause an assertion failure while writing the identifier table.
module ModB {
  header "ModBFile.h"
  export *
}

//--- ModAFile1.h
#include "shared.h"

//--- ModAFile2.h
// Including this file, triggers loading of the module ModA with nodes coming ModAFile1.h being hidden.

//--- ModBFile.h
// ModBFile depends on ModAFile2.h only.
#include "ModAFile2.h"
// Including shared.h here causes Sema to merge the AST nodes from shared.h with the hidden ones from ModA.
#include "shared.h"


//--- source.m
#include "ModBFile.h"

int main() { return MyVal_A + MyVal_B + MyVal_C + MyVal_D + MyVal_E; }
