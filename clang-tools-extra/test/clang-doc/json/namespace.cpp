// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html -check-prefix=HTML-CHECK

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
// CHECK-NEXT:    "DocumentationFileName": "index",
// CHECK-NEXT:    "Enums": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "InfoType": "enum",
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:          "LineNumber": 15
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
// CHECK-NEXT:            "End": true,
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
// CHECK-NEXT:       "End": true,
// CHECK-NEXT:       "InfoType": "function",
// CHECK-NEXT:       "IsStatic": false,
// CHECK-NEXT:       "Name": "myFunction",
// CHECK-NEXT:       "Params": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "End": true,
// CHECK-NEXT:           "Name": "Param",
// CHECK-NEXT:           "Type": {
// CHECK-NEXT:             "Name": "int",
// CHECK-NEXT:             "QualName": "int",
// CHECK-NEXT:             "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:           }
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
// CHECK-NEXT:   "HasEnums": true,
// CHECK-NEXT:   "HasFunctions": true,
// CHECK-NEXT:   "HasNamespaces": true,
// CHECK-NEXT:   "HasRecords": true,
// CHECK-NEXT:   "HasTypedefs": true,
// CHECK-NEXT:   "HasVariables": true,
// CHECK-NEXT:   "InfoType": "namespace",
// CHECK-NEXT:   "Name": "Global Namespace",
// CHECK-NEXT:   "Namespaces": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "End": true,
// CHECK-NEXT:       "Name": "NestedNamespace",
// CHECK-NEXT:       "QualName": "NestedNamespace",
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Records": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "DocumentationFileName": "_ZTV7MyClass",
// CHECK-NEXT:       "End": true,
// CHECK-NEXT:       "Name": "MyClass",
// CHECK-NEXT:       "Path": "GlobalNamespace",
// CHECK-NEXT:       "QualName": "MyClass",
// CHECK-NEXT:       "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Typedefs": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "End": true,
// CHECK-NEXT:      "InfoType": "typedef",
// CHECK-NEXT:      "IsUsing": false,
// CHECK-NEXT:      "Location": {
// CHECK-NEXT:        "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:        "LineNumber": 21
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
// CHECK-NEXT:       "End": true,
// CHECK-NEXT:       "InfoType": "variable",
// CHECK-NEXT:       "IsStatic": true,
// CHECK-NEXT:       "Location": {
// CHECK-NEXT:         "Filename": "{{.*}}namespace.cpp",
// CHECK-NEXT:         "LineNumber": 13
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

// HTML-CHECK:      <section id="Variables" class="section-container">
// HTML-CHECK-NEXT:     <h2>Variables</h2>
// HTML-CHECK-NEXT:     <div>
// HTML-CHECK-NEXT:         <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-CHECK-NEXT:             <pre><code class="language-cpp code-clang-doc" >static int Global</code></pre>
// HTML-CHECK-NEXT:             <p>Defined at line 13 of file {{.*}}namespace.cpp</p>
// HTML-CHECK-NEXT:         </div>
// HTML-CHECK-NEXT:     </div>
// HTML-CHECK-NEXT: </section>
