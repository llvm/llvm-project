// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.json

struct Foo;

// This is a nice class.
// It has some nice methods and fields.
// @brief This is a brief description.
struct MyClass {
  int PublicField;

  int myMethod(int MyParam);
  static void staticMethod();
  const int& getConst();
  
  enum Color {
    RED,
    GREEN,
    BLUE = 5
  };
  
  typedef int MyTypedef;
  
  class NestedClass;
protected:
  int protectedMethod();

  int ProtectedField;
};

// CHECK:       {
// CHECK-NEXT:    "Description": [
// CHECK-NEXT:      {
// CHECK-NEXT:       "FullComment": {
// CHECK-NEXT:         "Children": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "ParagraphComment": {
// CHECK-NEXT:               "Children": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "TextComment": " This is a nice class."
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "TextComment": " It has some nice methods and fields."
// CHECK-NEXT:                 },
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "TextComment": ""
// CHECK-NEXT:                 }
// CHECK-NEXT:               ]
// CHECK:               {
// CHECK-NEXT:             "BlockCommandComment": {
// CHECK-NEXT:               "Children": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "ParagraphComment": {
// CHECK-NEXT:                     "Children": [
// CHECK-NEXT:                       { 
// CHECK-NEXT:                         "TextComment": " This is a brief description." 
// CHECK-NEXT:                       }
// CHECK:                   "Command": "brief"
// CHECK:         "Enums": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:          "LineNumber": 17
// CHECK-NEXT:        },
// CHECK-NEXT:        "Members": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "Name": "RED",
// CHECK-NEXT:            "Value": "0"
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "Name": "GREEN",
// CHECK-NEXT:            "Value": "1"
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "Name": "BLUE",
// CHECK-NEXT:            "ValueExpr": "5"
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Name": "Color",
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Scoped": false,
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// COM:           FIXME: FullName is not emitted correctly.
// CHECK-NEXT:    "FullName": "",
// CHECK-NEXT:    "IsTypedef": false,
// CHECK-NEXT:    "Location": {
// CHECK-NEXT:      "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:      "LineNumber": 10
// CHECK-NEXT:    },
// CHECK-NEXT:    "Name": "MyClass",
// CHECK-NEXT:    "Namespace": [
// CHECK-NEXT:      "GlobalNamespace"
// CHECK-NEXT:    ],
// CHECK-NEXT:   "Path": "GlobalNamespace",
// CHECK-NEXT:   "ProtectedFunctions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "IsStatic": false,
// CHECK-NEXT:       "Name": "protectedMethod",
// CHECK-NEXT:       "Namespace": [
// CHECK-NEXT:         "MyClass",
// CHECK-NEXT:         "GlobalNamespace"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "ReturnType": {
// CHECK-NEXT:         "IsBuiltIn": false,
// CHECK-NEXT:         "IsTemplate": false,
// CHECK-NEXT:         "Name": "int",
// CHECK-NEXT:         "QualName": "int",
// CHECK-NEXT:         "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:       },
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "ProtectedMembers": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Name": "ProtectedField",
// CHECK-NEXT:        "Type": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "PublicFunctions": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "IsStatic": false,
// CHECK-NEXT:        "Name": "myMethod",
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Params": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "Name": "MyParam",
// CHECK-NEXT:            "Type": "int"
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "ReturnType": {
// CHECK-NEXT:          "IsBuiltIn": false,
// CHECK-NEXT:          "IsTemplate": false,
// CHECK-NEXT:          "Name": "int",
// CHECK-NEXT:          "QualName": "int",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      },
// CHECK:             "IsStatic": true,
// CHECK:             "Name": "getConst",
// CHECK:             "ReturnType": {
// CHECK-NEXT:          "IsBuiltIn": false,
// CHECK-NEXT:          "IsTemplate": false,
// CHECK-NEXT:          "Name": "const int &",
// CHECK-NEXT:          "QualName": "const int &",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK:         "PublicMembers": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Name": "PublicField",
// CHECK-NEXT:        "Type": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Records": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Name": "NestedClass",
// CHECK-NEXT:        "Path": "GlobalNamespace{{[\/]+}}MyClass",
// CHECK-NEXT:        "QualName": "NestedClass",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK-NEXT:    "TagType": "struct",
// CHECK-NEXT:    "Typedefs": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "IsUsing": false,
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:          "LineNumber": 23
// CHECK-NEXT:        },
// CHECK-NEXT:        "Name": "MyTypedef",
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "TypeDeclaration": "",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}",
// CHECK-NEXT:        "Underlying": {
// CHECK-NEXT:          "IsBuiltIn": false,
// CHECK-NEXT:          "IsTemplate": false,
// CHECK-NEXT:          "Name": "int",
// CHECK-NEXT:          "QualName": "int",
// CHECK-NEXT:          "USR": "0000000000000000000000000000000000000000"
// CHECK:         "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:  }
