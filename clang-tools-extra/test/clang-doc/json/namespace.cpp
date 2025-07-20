// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/index.json

class MyClass {};

void myFunction(int Param);

namespace NestedNamespace {
} // namespace NestedNamespace

static int Global;

enum Color {
  RED,
  GREEN,
  BLUE = 5
};

typedef int MyTypedef;

// CHECK:       { 
// CHECK-NEXT:    "Enums": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:          "LineNumber": 14
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
// CHECK-NEXT:        "Scoped": false,
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:   "Functions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "IsStatic": false,
// CHECK-NEXT:       "Name": "myFunction",
// CHECK-NEXT:       "Params": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "Name": "Param",
// CHECK-NEXT:           "Type": "int"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "ReturnType": {
// CHECK-NEXT:         "IsBuiltIn": true,
// CHECK-NEXT:         "IsTemplate": false,
// CHECK-NEXT:         "Name": "void",
// CHECK-NEXT:         "QualName": "void",
// CHECK-NEXT:         "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:       },
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Name": "",
// CHECK-NEXT:   "Namespaces": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "Name": "NestedNamespace",
// CHECK-NEXT:       "Path": "",
// CHECK-NEXT:       "QualName": "NestedNamespace",
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Records": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "Name": "MyClass",
// CHECK-NEXT:       "Path": "GlobalNamespace",
// CHECK-NEXT:       "QualName": "MyClass",
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Typedefs": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "IsUsing": false,
// CHECK-NEXT:      "Location": {
// CHECK-NEXT:        "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:        "LineNumber": 20
// CHECK-NEXT:      },
// CHECK-NEXT:      "Name": "MyTypedef",
// CHECK-NEXT:      "TypeDeclaration": "",
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}",
// CHECK-NEXT:      "Underlying": {
// CHECK-NEXT:        "IsBuiltIn": true,
// CHECK-NEXT:        "IsTemplate": false,
// CHECK-NEXT:        "Name": "int",
// CHECK-NEXT:        "QualName": "int",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:   "Variables": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "IsStatic": true,
// CHECK-NEXT:       "Location": {
// CHECK-NEXT:         "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:         "LineNumber": 12
// CHECK-NEXT:       },
// CHECK-NEXT:       "Name": "Global",
// CHECK-NEXT:       "Type": {
// CHECK-NEXT:         "IsBuiltIn": true,
// CHECK-NEXT:         "IsTemplate": false,
// CHECK-NEXT:         "Name": "int",
// CHECK-NEXT:         "QualName": "int",
// CHECK-NEXT:         "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:       },
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:  }
