// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs
// RUN: cat %t/docs/index.yaml | FileCheck %s --check-prefix=YAML

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs --format=md
// RUN: cat %t/docs/GlobalNamespace/index.md | FileCheck %s --check-prefix=MD

// RUN: clang-doc --doxygen --executor=standalone %s -output=%t/docs --format=html
// RUN: cat %t/docs/json/GlobalNamespace/index.json | FileCheck %s --check-prefix=JSON
// RUN: cat %t/docs/html/GlobalNamespace/_ZTV5tuple.html | FileCheck %s --check-prefix=HTML-STRUCT
// RUN: cat %t/docs/html/GlobalNamespace/index.html | FileCheck %s --check-prefix=HTML

// YAML: ---
// YAML-NEXT: USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT: ChildRecords:
// YAML-NEXT:   - Type:            Record
// YAML-NEXT:     Name:            'tuple'
// YAML-NEXT:     QualName:        'tuple'
// YAML-NEXT:     USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Path:            'GlobalNamespace'

// MD: # Global Namespace
// MD: ## Functions

template <class... T>
void ParamPackFunction(T... args);

// YAML-NEXT: ChildFunctions:
// YAML-NEXT:  - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'ParamPackFunction'
// YAML-NEXT:    Location:
// YAML-NEXT:      - LineNumber:      [[# @LINE - 6]]
// YAML-NEXT:        Filename:        '{{.*}}'
// YAML-NEXT:    Params:
// YAML-NEXT:      - Type:
// YAML-NEXT:          Name:            'T...'
// YAML-NEXT:          QualName:        'T...'
// YAML-NEXT:        Name:            'args'
// YAML-NEXT:    ReturnType:
// YAML-NEXT:      Type:
// YAML-NEXT:        Name:            'void'
// YAML-NEXT:        QualName:        'void'
// YAML-NEXT:    Template:
// YAML-NEXT:      Params:
// YAML-NEXT:        - Contents:        'class... T'

// MD: ### ParamPackFunction
// MD: *void ParamPackFunction(T... args)*

// JSON:           "Name": "ParamPackFunction",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "args",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "T...",
// JSON-NEXT:            "QualName": "T...",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Parameters": [
// JSON-NEXT:          {
// JSON-NEXT:            "End": true, 
// JSON-NEXT:            "Param": "class... T"
// JSON-NEXT:          }
// JSON-NEXT:        ],
// JSON-NEXT:        "VerticalDisplay": false
// JSON-NEXT:      },

// HTML:        <pre><code class="language-cpp code-clang-doc">template &lt;class... T</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:   <pre><code class="language-cpp code-clang-doc">void ParamPackFunction (T... args)</code></pre>

template <typename T, int U = 1>
void function(T x) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 5]]
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'T'
// YAML-NEXT:           QualName:        'T'
// YAML-NEXT:         Name:            'x'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:       Params:
// YAML-NEXT:         - Contents:        'typename T'
// YAML-NEXT:         - Contents:        'int U = 1'

// MD: ### function
// MD: *void function(T x)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 23]]*

// JSON:           "Name": "function",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "x",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "T",
// JSON-NEXT:            "QualName": "T",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Parameters": [
// JSON-NEXT:          {
// JSON-NEXT:            "Param": "typename T"
// JSON-NEXT:          },
// JSON-NEXT:          {
// JSON-NEXT:            "End": true, 
// JSON-NEXT:            "Param": "int U = 1"
// JSON-NEXT:          }
// JSON-NEXT:        ],
// JSON-NEXT:        "VerticalDisplay": false
// JSON-NEXT:      }

// HTML:           <pre><code class="language-cpp code-clang-doc">template &lt;typename T, int U = 1</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:      <pre><code class="language-cpp code-clang-doc">void function (T x)</code></pre>
// HTML-NEXT:      <p>Defined at line [[# @LINE - 59]] of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

template <typename A, typename B, typename C, typename D, typename E>
void longFunction(A a, B b, C c, D d, E e) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'longFunction'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 5]]
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'A'
// YAML-NEXT:           QualName:        'A'
// YAML-NEXT:         Name:            'a'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'B'
// YAML-NEXT:           QualName:        'B'
// YAML-NEXT:         Name:            'b'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'C'
// YAML-NEXT:           QualName:        'C'
// YAML-NEXT:         Name:            'c'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'D'
// YAML-NEXT:           QualName:        'D'
// YAML-NEXT:         Name:            'd'
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'E'
// YAML-NEXT:           QualName:        'E'
// YAML-NEXT:         Name:            'e'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:         Params:
// YAML-NEXT:           - Contents:        'typename A'
// YAML-NEXT:           - Contents:        'typename B'
// YAML-NEXT:           - Contents:        'typename C'
// YAML-NEXT:           - Contents:        'typename D'
// YAML-NEXT:           - Contents:        'typename E'

// MD: ### longFunction
// MD: *void longFunction(A a, B b, C c, D d, E e)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 42]]*

// JSON:           "Name": "longFunction",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "a",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "A",
// JSON-NEXT:            "QualName": "A",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "b",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "B",
// JSON-NEXT:            "QualName": "B",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "c",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "C",
// JSON-NEXT:            "QualName": "C",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:       {
// JSON-NEXT:          "Name": "d",
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "D",
// JSON-NEXT:            "QualName": "D",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "e",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "E",
// JSON-NEXT:            "QualName": "E",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:          "Parameters": [
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename A"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename B"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename C"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "typename D"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "End": true,
// JSON-NEXT:              "Param": "typename E"
// JSON-NEXT:            }
// JSON-NEXT:          ],
// JSON-NEXT:          "VerticalDisplay": true
// JSON-NEXT:        },
// JSON-NEXT:        "USR": "{{([0-9A-F]{40})}}",
// JSON-NEXT:        "VerticalDisplay": true
// JSON-NEXT:      }

// HTML:           <pre>
// HTML-SAME:        <code class="language-cpp code-clang-doc">template &lt;</code>
// HTML-SAME:        <span class="param-container">
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename A, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename B, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename C, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename D, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">typename E</code></span>
// HTML-SAME:        </span>
// HTML-SAME:        <code class="language-cpp code-clang-doc">&gt;</code>
// HTML-SAME:      </pre>
// HTML-NEXT:      <pre>
// HTML-SAME:        <code class="language-cpp code-clang-doc">void longFunction (</code>
// HTML-SAME:        <span class="param-container">
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">A</code> <code class="language-cpp code-clang-doc">a, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">B</code> <code class="language-cpp code-clang-doc">b, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">C</code> <code class="language-cpp code-clang-doc">c, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">D</code> <code class="language-cpp code-clang-doc">d, </code></span>
// HTML-SAME:          <span class="param"><code class="language-cpp code-clang-doc">E</code> <code class="language-cpp code-clang-doc">e</code></span>
// HTML-SAME:        </span>
// HTML-SAME:        <code class="language-cpp code-clang-doc">)</code>
// HTML-SAME:      </pre>
// HTML-NEXT:      <p>Defined at line [[# @LINE - 142]] of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

template <>
void function<bool, 0>(bool x) {}

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:     Name:            'function'
// YAML-NEXT:     DefLocation:
// YAML-NEXT:       LineNumber:      [[# @LINE - 6]]
// YAML-NEXT:       Filename:        '{{.*}}'
// YAML-NEXT:     Params:
// YAML-NEXT:       - Type:
// YAML-NEXT:           Name:            'bool'
// YAML-NEXT:           QualName:        'bool'
// YAML-NEXT:         Name:            'x'
// YAML-NEXT:     ReturnType:
// YAML-NEXT:       Type:
// YAML-NEXT:         Name:            'void'
// YAML-NEXT:         QualName:        'void'
// YAML-NEXT:     Template:
// YAML-NEXT:       Specialization:
// YAML-NEXT:         SpecializationOf: '{{([0-9A-F]{40})}}'
// YAML-NEXT:         Params:
// YAML-NEXT:           - Contents:        'bool'
// YAML-NEXT:           - Contents:        '0'

// MD: ### function
// MD: *void function(bool x)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 26]]*

// JSON:           "Name": "function",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "x",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "bool",
// JSON-NEXT:            "QualName": "bool",
// JSON-NEXT:            "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": true,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "void",
// JSON-NEXT:        "QualName": "void",
// JSON-NEXT:        "USR": "0000000000000000000000000000000000000000"
// JSON-NEXT:      },
// JSON-NEXT:      "Template": {
// JSON-NEXT:        "Specialization": {
// JSON-NEXT:          "Parameters": [
// JSON-NEXT:            {
// JSON-NEXT:              "Param": "bool"
// JSON-NEXT:            },
// JSON-NEXT:            {
// JSON-NEXT:              "End": true, 
// JSON-NEXT:              "Param": "0"
// JSON-NEXT:            }
// JSON-NEXT:          ],
// JSON-NEXT:          "SpecializationOf": "{{([0-9A-F]{40})}}",
// JSON-NEXT:          "VerticalDisplay": false
// JSON-NEXT:        }
// JSON-NEXT:      },

// HTML:           <pre><code class="language-cpp code-clang-doc">template &lt;</code><code class="language-cpp code-clang-doc">&gt;</code></pre>
// HTML-NEXT:      <pre><code class="language-cpp code-clang-doc">void function&lt;bool, 0&gt; (bool x)</code></pre>
// HTML-NEXT:      <p>Defined at line [[# @LINE - 65]] of file {{.*}}templates.cpp</p>
// HTML-NEXT:  </div>

/// A Tuple type
///
/// Does Tuple things.
template <typename... Tys>
struct tuple {};

// HTML-STRUCT:        <section class="hero section-container">
// HTML-STRUCT-NEXT:       <pre><code class="language-cpp code-clang-doc">template &lt;typename... Tys&gt;</code></pre>
// HTML-STRUCT-NEXT:       <div class="hero__title">
// HTML-STRUCT-NEXT:           <h1 class="hero__title-large">struct tuple</h1>
// HTML-STRUCT-NEXT:           <p>Defined at line [[# @LINE - 6]] of file {{.*}}templates.cpp</p>
// HTML-STRUCT-NEXT:           <div class="doc-card">
// HTML-STRUCT-NEXT:               <div class="nested-delimiter-container">
// HTML-STRUCT-NEXT:                   <p> A Tuple type</p>
// HTML-STRUCT-NEXT:               </div>
// HTML-STRUCT-NEXT:               <div class="nested-delimiter-container">
// HTML-STRUCT-NEXT:                   <p> Does Tuple things.</p>
// HTML-STRUCT-NEXT:               </div>
// HTML-STRUCT-NEXT:           </div>
// HTML-STRUCT-NEXT:       </div>
// HTML-STRUCT-NEXT:   </section>

/// A function with a tuple parameter
///
/// \param t The input to func_with_tuple_param
tuple<int, int, bool> func_with_tuple_param(tuple<int, int, bool> t) { return t; }

// YAML-NEXT:   - USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:    Name:            'func_with_tuple_param'
// YAML-NEXT:    Description:
// YAML-NEXT:      - Kind:            FullComment
// YAML-NEXT:        Children:
// YAML-NEXT:          - Kind:            ParagraphComment
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            TextComment
// YAML-NEXT:                Text:            ' A function with a tuple parameter'
// YAML-NEXT:          - Kind:            ParagraphComment
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            TextComment
// YAML-NEXT:          - Kind:            ParamCommandComment
// YAML-NEXT:            Direction:       '[in]'
// YAML-NEXT:            ParamName:       't'
// YAML-NEXT:            Children:
// YAML-NEXT:              - Kind:            ParagraphComment
// YAML-NEXT:                Children:
// YAML-NEXT:                  - Kind:            TextComment
// YAML-NEXT:                    Text:            ' The input to func_with_tuple_param'
// YAML-NEXT:    DefLocation:
// YAML-NEXT:      LineNumber:      [[# @LINE - 23]]
// YAML-NEXT:      Filename:
// YAML-NEXT:    Params:
// YAML-NEXT:      - Type:
// YAML-NEXT:          Type:            Record
// YAML-NEXT:          Name:            'tuple'
// YAML-NEXT:          QualName:        'tuple<int, int, bool>'
// YAML-NEXT:          USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:          Path:            'GlobalNamespace'
// YAML-NEXT:        Name:            't'
// YAML-NEXT:    ReturnType:
// YAML-NEXT:      Type:
// YAML-NEXT:        Type:            Record
// YAML-NEXT:        Name:            'tuple'
// YAML-NEXT:        QualName:        'tuple<int, int, bool>'
// YAML-NEXT:        USR:             '{{([0-9A-F]{40})}}'
// YAML-NEXT:        Path:            'GlobalNamespace'
// YAML-NEXT: ...

// MD: ### func_with_tuple_param
// MD: *tuple<int, int, bool> func_with_tuple_param(tuple<int, int, bool> t)*
// MD: *Defined at {{.*}}templates.cpp#[[# @LINE - 44]]*
// MD:  A function with a tuple parameter
// MD: **t** The input to func_with_tuple_param

// JSON:           "Name": "func_with_tuple_param",
// JSON-NEXT:      "Params": [
// JSON-NEXT:        {
// JSON-NEXT:          "Name": "t",
// JSON-NEXT:          "ParamEnd": true,
// JSON-NEXT:          "Type": {
// JSON-NEXT:            "Name": "tuple",
// JSON-NEXT:            "Path": "GlobalNamespace",
// JSON-NEXT:            "QualName": "tuple<int, int, bool>",
// JSON-NEXT:            "USR": "{{([0-9A-F]{40})}}"
// JSON-NEXT:          }
// JSON-NEXT:        }
// JSON-NEXT:      ],
// JSON-NEXT:      "ReturnType": {
// JSON-NEXT:        "IsBuiltIn": false,
// JSON-NEXT:        "IsTemplate": false,
// JSON-NEXT:        "Name": "tuple",
// JSON-NEXT:        "QualName": "tuple<int, int, bool>",
// JSON-NEXT:        "USR": "{{([0-9A-F]{40})}}"
// JSON-NEXT:      },

// HTML:           <pre><code class="language-cpp code-clang-doc">tuple&lt;int, int, bool&gt; func_with_tuple_param (tuple&lt;int, int, bool&gt; t)</code></pre>
// HTML-NEXT:      <div class="doc-card">
// HTML-NEXT:          <div class="nested-delimiter-container">
// HTML-NEXT:              <p> A function with a tuple parameter</p>
// HTML-NEXT:          </div>
// HTML-NEXT:          <div class="nested-delimiter-container">
// HTML-NEXT:              <h3>Parameters</h3>
// HTML-NEXT:              <div>
// HTML-NEXT:                  <b>t</b>   The input to func_with_tuple_param
// HTML-NEXT:              </div> 
// HTML-NEXT:          </div>
// HTML-NEXT:      </div>
// HTML-NEXT:      <p>Defined at line [[# @LINE - 81]] of file {{.*}}templates.cpp</p>
