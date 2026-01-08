// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClass.json
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html -check-prefix=HTML

/// This is a struct friend.
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
  
  friend struct Foo;
  /// This is a function template friend.
  template<typename T> friend void friendFunction(int);
protected:
  int protectedMethod();

  int ProtectedField;
private:
  int PrivateField;
};

// CHECK:       {
// CHECK-NEXT:    "Contexts": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "DocumentationFileName": "index",
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "Name": "Global Namespace",
// CHECK-NEXT:        "QualName": "GlobalNamespace",
// CHECK-NEXT:        "RelativePath": "./",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Description": {
// CHECK-NEXT:      "BriefComments": [
// CHECK-NEXT:        [
// CHECK-NEXT:          {
// CHECK-NEXT:            "TextComment": " This is a brief description."
// CHECK:           "HasBriefComments": true,
// CHECK-NEXT:      "HasParagraphComments": true,
// CHECK-NEXT:      "ParagraphComments": [
// CHECK-NEXT:        [
// CHECK-NEXT:          {
// CHECK-NEXT:            "TextComment": " This is a nice class."
// CHECK-NEXT:          },
// CHECK-NEXT:          {
// CHECK-NEXT:            "TextComment": " It has some nice methods and fields."
// CHECK-NEXT:          }
// CHECK:         "DocumentationFileName": "_ZTV7MyClass",
// CHECK:         "Enums": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "InfoType": "enum",
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:          "LineNumber": 19
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
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Scoped": false,
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Friends": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Description": {
// CHECK-NEXT:          "HasParagraphComments": true,
// CHECK-NEXT:          "ParagraphComments": [
// CHECK-NEXT:            [
// CHECK-NEXT:              {
// CHECK-NEXT:                "TextComment": " This is a function template friend."
// CHECK-NEXT:              }
// CHECK-NEXT:            ]
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        "InfoType": "friend",
// CHECK-NEXT:        "IsClass": false,
// CHECK-NEXT:        "Params": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "End": true,
// CHECK-NEXT:            "Name": "",
// CHECK-NEXT:            "Type": {
// CHECK-NEXT:              "Name": "int",
// CHECK-NEXT:              "QualName": "int",
// CHECK-NEXT:              "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Reference": {
// CHECK-NEXT:          "Name": "friendFunction",
// CHECK-NEXT:          "QualName": "friendFunction",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        "ReturnType": {
// CHECK-NEXT:          "IsBuiltIn": true,
// CHECK-NEXT:          "IsTemplate": false,
// CHECK-NEXT:          "Name": "void",
// CHECK-NEXT:          "QualName": "void",
// CHECK-NEXT:          "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:        },
// CHECK-NEXT:        "Template": {
// CHECK-NEXT:          "Parameters": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "End": true,
// CHECK-NEXT:              "Param": "typename T"
// CHECK-NEXT:            }
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      },
// CHECK-NEXT:      {
// CHECK-NEXT:        "Description": {
// CHECK-NEXT:          "HasParagraphComments": true,
// CHECK-NEXT:          "ParagraphComments": [
// CHECK-NEXT:            [
// CHECK-NEXT:              {
// CHECK-NEXT:                "TextComment": " This is a struct friend."
// CHECK-NEXT:              }
// CHECK-NEXT:            ]
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "InfoType": "friend",
// CHECK-NEXT:        "IsClass": true,
// CHECK-NEXT:        "Reference": {
// CHECK-NEXT:          "Name": "Foo",
// CHECK-NEXT:          "Path": "GlobalNamespace",
// CHECK-NEXT:          "QualName": "Foo",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "HasContexts": true,
// CHECK-NEXT:    "HasEnums": true,
// CHECK-NEXT:    "HasFriends": true,
// CHECK-NEXT:    "HasPrivateMembers": true,
// CHECK-NEXT:    "HasProtectedMembers": true,
// CHECK-NEXT:    "HasPublicFunctions": true,
// CHECK-NEXT:    "HasPublicMembers": true,
// CHECK-NEXT:    "HasRecords": true,
// CHECK-NEXT:    "HasTypedefs": true,
// CHECK-NEXT:    "InfoType": "record",
// CHECK-NEXT:    "IsTypedef": false,
// CHECK-NEXT:    "Location": {
// CHECK-NEXT:      "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:      "LineNumber": 12
// CHECK-NEXT:    },
// CHECK-NEXT:    "MangledName": "_ZTV7MyClass",
// CHECK-NEXT:    "Name": "MyClass",
// CHECK-NEXT:    "Namespace": [
// CHECK-NEXT:      "GlobalNamespace"
// CHECK-NEXT:    ],
// CHECK-NEXT:   "Path": "GlobalNamespace",
// CHECK-NEXT:   "PrivateMembers": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "IsStatic": false,
// CHECK-NEXT:       "Name": "PrivateField",
// CHECK-NEXT:       "Type": "int"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "ProtectedFunctions": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "InfoType": "function",
// CHECK-NEXT:       "IsStatic": false,
// CHECK-NEXT:       "Name": "protectedMethod",
// CHECK-NEXT:       "Namespace": [
// CHECK-NEXT:         "MyClass",
// CHECK-NEXT:         "GlobalNamespace"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "ReturnType": {
// CHECK-NEXT:         "IsBuiltIn": true,
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
// CHECK-NEXT:        "IsStatic": false,
// CHECK-NEXT:        "Name": "ProtectedField",
// CHECK-NEXT:        "Type": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "PublicFunctions": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "InfoType": "function",
// CHECK-NEXT:        "IsStatic": false,
// CHECK-NEXT:        "Name": "myMethod",
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Params": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "End": true,
// CHECK-NEXT:            "Name": "MyParam",
// CHECK-NEXT:            "Type": {
// CHECK-NEXT:              "Name": "int",
// CHECK-NEXT:              "QualName": "int",
// CHECK-NEXT:              "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "ReturnType": {
// CHECK-NEXT:          "IsBuiltIn": true,
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
// CHECK-NEXT:        "IsStatic": false,
// CHECK-NEXT:        "Name": "PublicField",
// CHECK-NEXT:        "Type": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Records": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "DocumentationFileName": "_ZTVN7MyClass11NestedClassE",
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "Name": "NestedClass",
// CHECK-NEXT:        "Path": "GlobalNamespace{{[\/]+}}MyClass",
// CHECK-NEXT:        "PathStem": "MyClass",
// CHECK-NEXT:        "QualName": "NestedClass",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ]
// CHECK-NEXT:    "TagType": "struct",
// CHECK-NEXT:    "Typedefs": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "InfoType": "typedef",
// CHECK-NEXT:        "IsUsing": false,
// CHECK-NEXT:        "Location": {
// CHECK-NEXT:          "Filename": "{{.*}}class.cpp",
// CHECK-NEXT:          "LineNumber": 25
// CHECK-NEXT:        },
// CHECK-NEXT:        "Name": "MyTypedef",
// CHECK-NEXT:        "Namespace": [
// CHECK-NEXT:          "MyClass",
// CHECK-NEXT:          "GlobalNamespace"
// CHECK-NEXT:        ],
// CHECK-NEXT:        "TypeDeclaration": "",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}",
// CHECK-NEXT:        "Underlying": {
// CHECK-NEXT:          "IsBuiltIn": true,
// CHECK-NEXT:          "IsTemplate": false,
// CHECK-NEXT:          "Name": "int",
// CHECK-NEXT:          "QualName": "int",
// CHECK-NEXT:          "USR": "0000000000000000000000000000000000000000"
// CHECK:         "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:  }

// HTML:              <a class="sidebar-item" href="#Classes">Inner Classes</a>
// HTML-NEXT:     </summary>
// HTML-NEXT:     <ul>
// HTML-NEXT:         <li class="sidebar-item-container">
// HTML-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">NestedClass</a>
// HTML-NEXT:         </li>
// HTML-NEXT:     </ul>
// HTML-NEXT: </details>
// HTML:              <a class="sidebar-item" href="#Friends">Friends</a>
// HTML-NEXT:     </summary>
// HTML-NEXT:     <ul>
// HTML-NEXT:         <li class="sidebar-item-container">
// HTML-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">friendFunction</a>
// HTML-NEXT:         </li>
// HTML-NEXT:         <li class="sidebar-item-container">
// HTML-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Foo</a>
// HTML-NEXT:         </li>
// HTML-NEXT:     </ul>
// HTML-NEXT: </details>
// HTML:      <section id="ProtectedMembers" class="section-container">
// HTML-NEXT:     <h2>Protected Members</h2>
// HTML-NEXT:     <div>
// HTML-NEXT:         <div id="ProtectedField" class="delimiter-container">
// HTML-NEXT:             <pre><code class="language-cpp code-clang-doc" >int ProtectedField</code></pre>
// HTML-NEXT:         </div>
// HTML-NEXT:     </div>
// HTML-NEXT: </section>
// HTML:      <section id="Classes" class="section-container">
// HTML-NEXT:     <h2>Inner Classes</h2>
// HTML-NEXT:     <ul class="class-container">
// HTML-NEXT:         <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-NEXT:             <a href="MyClass/_ZTVN7MyClass11NestedClassE.html">
// HTML-NEXT:                 <pre><code class="language-cpp code-clang-doc">class NestedClass</code></pre>
// HTML-NEXT:             </a>
// HTML-NEXT:         </li>
// HTML-NEXT:     </ul>
// HTML-NEXT: </section>
// HTML:      <section id="Friends" class="section-container">
// HTML-NEXT:     <h2>Friends</h2>
// HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-NEXT:         <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt;</code></pre>
// HTML-NEXT:         <pre><code class="language-cpp code-clang-doc">void MyClass (int )</code></pre>
// HTML-NEXT:         <div class="nested-delimiter-container">
// HTML-NEXT:             <p> This is a function template friend.</p>
// HTML-NEXT:         </div>
// HTML-NEXT:     </div>
// HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-NEXT:         <pre><code class="language-cpp code-clang-doc">class Foo</code></pre>
// HTML-NEXT:         <div class="nested-delimiter-container">
// HTML-NEXT:             <p> This is a struct friend.</p>
// HTML-NEXT:         </div>
// HTML-NEXT:     </div>
// HTML-NEXT: </section>
