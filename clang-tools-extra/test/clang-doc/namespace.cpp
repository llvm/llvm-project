// RUN: clang-doc --format=html --output=%t/docs --executor=standalone %s
// RUN: clang-doc --format=md --output=%t/docs --executor=standalone %s
// RUN: FileCheck %s -input-file=%t/docs/index_json.js -check-prefix=JSON-INDEX
// RUN: FileCheck %s -input-file=%t/docs/@nonymous_namespace/AnonClass.html -check-prefix=HTML-ANON-CLASS
// RUN: FileCheck %s -input-file=%t/docs/@nonymous_namespace/index.html -check-prefix=HTML-ANON-INDEX
// RUN: FileCheck %s -input-file=%t/docs/AnotherNamespace/ClassInAnotherNamespace.html -check-prefix=HTML-ANOTHER-CLASS
// RUN: FileCheck %s -input-file=%t/docs/AnotherNamespace/index.html -check-prefix=HTML-ANOTHER-INDEX
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/index.html -check-prefix=HTML-GLOBAL-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.html -check-prefix=HTML-NESTED-CLASS
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/NestedNamespace/index.html -check-prefix=HTML-NESTED-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/index.html -check-prefix=HTML-PRIMARY-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/ClassInPrimaryNamespace.html -check-prefix=HTML-PRIMARY-CLASS
// RUN: FileCheck %s -input-file=%t/docs/@nonymous_namespace/AnonClass.md -check-prefix=MD-ANON-CLASS
// RUN: FileCheck %s -input-file=%t/docs/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX
// RUN: FileCheck %s -input-file=%t/docs/AnotherNamespace/ClassInAnotherNamespace.md -check-prefix=MD-ANOTHER-CLASS
// RUN: FileCheck %s -input-file=%t/docs/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX
// RUN: FileCheck %s -input-file=%t/docs/GlobalNamespace/index.md -check-prefix=MD-GLOBAL-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.md -check-prefix=MD-NESTED-CLASS
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX
// RUN: FileCheck %s -input-file=%t/docs/PrimaryNamespace/ClassInPrimaryNamespace.md -check-prefix=MD-PRIMARY-CLASS
// RUN: FileCheck %s -input-file=%t/docs/all_files.md -check-prefix=MD-ALL-FILES
// RUN: FileCheck %s -input-file=%t/docs/index.md -check-prefix=MD-INDEX

// Anonymous Namespace
namespace
{
    void anonFunction() {}
    class AnonClass {};
}

// Primary Namespace
namespace PrimaryNamespace {
    // Function in PrimaryNamespace
    void functionInPrimaryNamespace() {}

    // Class in PrimaryNamespace
    class ClassInPrimaryNamespace {};

    // Nested namespace
    namespace NestedNamespace {
        // Function in NestedNamespace
        void functionInNestedNamespace() {}
        // Class in NestedNamespace
        class ClassInNestedNamespace {};
    }
}

// AnotherNamespace
namespace AnotherNamespace {
    // Function in AnotherNamespace
    void functionInAnotherNamespace() {}
    // Class in AnotherNamespace
    class ClassInAnotherNamespace {};
}

// JSON-INDEX: async function LoadIndex() {
// JSON-INDEX-NEXT: return{
// JSON-INDEX-NEXT:   "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:   "Name": "",
// JSON-INDEX-NEXT:   "RefType": "default",
// JSON-INDEX-NEXT:   "Path": "",
// JSON-INDEX-NEXT:   "Children": [
// JSON-INDEX-NEXT:     {
// JSON-INDEX-NEXT:       "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:       "Name": "@nonymous_namespace",
// JSON-INDEX-NEXT:       "RefType": "namespace",
// JSON-INDEX-NEXT:       "Path": "@nonymous_namespace",
// JSON-INDEX-NEXT:       "Children": [
// JSON-INDEX-NEXT:         {
// JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:           "Name": "AnonClass",
// JSON-INDEX-NEXT:           "RefType": "record",
// JSON-INDEX-NEXT:           "Path": "@nonymous_namespace",
// JSON-INDEX-NEXT:           "Children": []
// JSON-INDEX-NEXT:         }
// JSON-INDEX-NEXT:       ]
// JSON-INDEX-NEXT:     },
// JSON-INDEX-NEXT:     {
// JSON-INDEX-NEXT:       "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:       "Name": "AnotherNamespace",
// JSON-INDEX-NEXT:       "RefType": "namespace",
// JSON-INDEX-NEXT:       "Path": "AnotherNamespace",
// JSON-INDEX-NEXT:       "Children": [
// JSON-INDEX-NEXT:         {
// JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:           "Name": "ClassInAnotherNamespace",
// JSON-INDEX-NEXT:           "RefType": "record",
// JSON-INDEX-NEXT:           "Path": "AnotherNamespace",
// JSON-INDEX-NEXT:           "Children": []
// JSON-INDEX-NEXT:         }
// JSON-INDEX-NEXT:       ]
// JSON-INDEX-NEXT:     },
// JSON-INDEX-NEXT:     {
// JSON-INDEX-NEXT:       "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:       "Name": "GlobalNamespace",
// JSON-INDEX-NEXT:       "RefType": "namespace",
// JSON-INDEX-NEXT:       "Path": "GlobalNamespace",
// JSON-INDEX-NEXT:       "Children": []
// JSON-INDEX-NEXT:     },
// JSON-INDEX-NEXT:     {
// JSON-INDEX-NEXT:       "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:       "Name": "PrimaryNamespace",
// JSON-INDEX-NEXT:       "RefType": "namespace",
// JSON-INDEX-NEXT:       "Path": "PrimaryNamespace",
// JSON-INDEX-NEXT:       "Children": [
// JSON-INDEX-NEXT:         {
// JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:           "Name": "ClassInPrimaryNamespace",
// JSON-INDEX-NEXT:           "RefType": "record",
// JSON-INDEX-NEXT:           "Path": "PrimaryNamespace",
// JSON-INDEX-NEXT:           "Children": []
// JSON-INDEX-NEXT:         },
// JSON-INDEX-NEXT:         {
// JSON-INDEX-NEXT:           "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:           "Name": "NestedNamespace",
// JSON-INDEX-NEXT:           "RefType": "namespace",
// JSON-INDEX-NEXT:           "Path": "PrimaryNamespace{{[\/]+}}NestedNamespace",
// JSON-INDEX-NEXT:           "Children": [
// JSON-INDEX-NEXT:             {
// JSON-INDEX-NEXT:               "USR": "{{([0-9A-F]{40})}}",
// JSON-INDEX-NEXT:               "Name": "ClassInNestedNamespace",
// JSON-INDEX-NEXT:               "RefType": "record",
// JSON-INDEX-NEXT:               "Path": "PrimaryNamespace{{[\/]+}}NestedNamespace",
// JSON-INDEX-NEXT:               "Children": []
// JSON-INDEX-NEXT:             }
// JSON-INDEX-NEXT:           ]
// JSON-INDEX-NEXT:         }
// JSON-INDEX-NEXT:       ]
// JSON-INDEX-NEXT:     }
// JSON-INDEX-NEXT:   ]
// JSON-INDEX-NEXT: };
// JSON-INDEX-NEXT: }

// HTML-ANON-CLASS: <!DOCTYPE html>
// HTML-ANON-CLASS-NEXT: <meta charset="utf-8"/>
// HTML-ANON-CLASS-NEXT: <title>class AnonClass</title>
// HTML-ANON-CLASS-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-ANON-CLASS-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-ANON-CLASS-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-ANON-CLASS-NEXT: <header id="project-title"></header>
// HTML-ANON-CLASS-NEXT: <main>
// HTML-ANON-CLASS-NEXT:   <div id="sidebar-left" path="@nonymous_namespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-ANON-CLASS-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-ANON-CLASS-NEXT:     <h1>class AnonClass</h1>
// HTML-ANON-CLASS-NEXT:     <p>Defined at line 31 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-ANON-CLASS-NEXT:   </div>
// HTML-ANON-CLASS-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right"></div>
// HTML-ANON-CLASS-NEXT: </main>

// HTML-ANON-INDEX: <!DOCTYPE html>
// HTML-ANON-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-ANON-INDEX-NEXT: <title>namespace @nonymous_namespace</title>
// HTML-ANON-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-ANON-INDEX-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-ANON-INDEX-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-ANON-INDEX-NEXT: <header id="project-title"></header>
// HTML-ANON-INDEX-NEXT: <main>
// HTML-ANON-INDEX-NEXT:   <div id="sidebar-left" path="@nonymous_namespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-ANON-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-ANON-INDEX-NEXT:     <h1>namespace @nonymous_namespace</h1>
// HTML-ANON-INDEX-NEXT:     <div>
// HTML-ANON-INDEX-NEXT:       <div>
// HTML-ANON-INDEX-NEXT:        <p> Anonymous Namespace</p>
// HTML-ANON-INDEX-NEXT:       </div>
// HTML-ANON-INDEX-NEXT:     </div>
// HTML-ANON-INDEX-NEXT:     <h2 id="Records">Records</h2>
// HTML-ANON-INDEX-NEXT:     <ul>
// HTML-ANON-INDEX-NEXT:       <li>
// HTML-ANON-INDEX-NEXT:         <a href="AnonClass.html">AnonClass</a>
// HTML-ANON-INDEX-NEXT:       </li>
// HTML-ANON-INDEX-NEXT:     </ul>
// HTML-ANON-INDEX-NEXT:     <h2 id="Functions">Functions</h2>
// HTML-ANON-INDEX-NEXT:     <div>
// HTML-ANON-INDEX-NEXT:       <h3 id="{{([0-9A-F]{40})}}">anonFunction</h3>
// HTML-ANON-INDEX-NEXT:       <p>void anonFunction()</p>
// HTML-ANON-INDEX-NEXT:       <p>Defined at line 30 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-ANON-INDEX-NEXT:     </div>
// HTML-ANON-INDEX-NEXT:   </div>
// HTML-ANON-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-ANON-INDEX-NEXT:     <ol>
// HTML-ANON-INDEX-NEXT:       <li>
// HTML-ANON-INDEX-NEXT:         <span>
// HTML-ANON-INDEX-NEXT:           <a href="#Records">Records</a>
// HTML-ANON-INDEX-NEXT:         </span>
// HTML-ANON-INDEX-NEXT:       </li>
// HTML-ANON-INDEX-NEXT:       <li>
// HTML-ANON-INDEX-NEXT:         <span>
// HTML-ANON-INDEX-NEXT:           <a href="#Functions">Functions</a>
// HTML-ANON-INDEX-NEXT:         </span>
// HTML-ANON-INDEX-NEXT:         <ul>
// HTML-ANON-INDEX-NEXT:           <li>
// HTML-ANON-INDEX-NEXT:             <span>
// HTML-ANON-INDEX-NEXT:               <a href="#{{([0-9A-F]{40})}}">anonFunction</a>
// HTML-ANON-INDEX-NEXT:             </span>
// HTML-ANON-INDEX-NEXT:           </li>
// HTML-ANON-INDEX-NEXT:         </ul>
// HTML-ANON-INDEX-NEXT:       </li>
// HTML-ANON-INDEX-NEXT:     </ol>
// HTML-ANON-INDEX-NEXT:   </div>
// HTML-ANON-INDEX-NEXT: </main>

// HTML-ANOTHER-CLASS: <!DOCTYPE html>
// HTML-ANOTHER-CLASS-NEXT: <meta charset="utf-8"/>
// HTML-ANOTHER-CLASS-NEXT: <title>class ClassInAnotherNamespace</title>
// HTML-ANOTHER-CLASS-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-ANOTHER-CLASS-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-ANOTHER-CLASS-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-ANOTHER-CLASS-NEXT: <header id="project-title"></header>
// HTML-ANOTHER-CLASS-NEXT: <main>
// HTML-ANOTHER-CLASS-NEXT:   <div id="sidebar-left" path="AnotherNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-ANOTHER-CLASS-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-ANOTHER-CLASS-NEXT:     <h1>class ClassInAnotherNamespace</h1>
// HTML-ANOTHER-CLASS-NEXT:     <p>Defined at line 56 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-ANOTHER-CLASS-NEXT:     <div>
// HTML-ANOTHER-CLASS-NEXT:       <div>
// HTML-ANOTHER-CLASS-NEXT:         <p> Class in AnotherNamespace</p>
// HTML-ANOTHER-CLASS-NEXT:       </div>
// HTML-ANOTHER-CLASS-NEXT:     </div>
// HTML-ANOTHER-CLASS-NEXT:   </div>
// HTML-ANOTHER-CLASS-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right"></div>
// HTML-ANOTHER-CLASS-NEXT: </main>

// HTML-ANOTHER-INDEX: <!DOCTYPE html>
// HTML-ANOTHER-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-ANOTHER-INDEX-NEXT: <title>namespace AnotherNamespace</title>
// HTML-ANOTHER-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-ANOTHER-INDEX-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-ANOTHER-INDEX-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-ANOTHER-INDEX-NEXT: <header id="project-title"></header>
// HTML-ANOTHER-INDEX-NEXT: <main>
// HTML-ANOTHER-INDEX-NEXT:   <div id="sidebar-left" path="AnotherNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-ANOTHER-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-ANOTHER-INDEX-NEXT:     <h1>namespace AnotherNamespace</h1>
// HTML-ANOTHER-INDEX-NEXT:     <div>
// HTML-ANOTHER-INDEX-NEXT:       <div>
// HTML-ANOTHER-INDEX-NEXT:         <p> AnotherNamespace</p>
// HTML-ANOTHER-INDEX-NEXT:       </div>
// HTML-ANOTHER-INDEX-NEXT:     </div>
// HTML-ANOTHER-INDEX-NEXT:     <h2 id="Records">Records</h2>
// HTML-ANOTHER-INDEX-NEXT:     <ul>
// HTML-ANOTHER-INDEX-NEXT:       <li>
// HTML-ANOTHER-INDEX-NEXT:         <a href="ClassInAnotherNamespace.html">ClassInAnotherNamespace</a>
// HTML-ANOTHER-INDEX-NEXT:       </li>
// HTML-ANOTHER-INDEX-NEXT:     </ul>
// HTML-ANOTHER-INDEX-NEXT:     <h2 id="Functions">Functions</h2>
// HTML-ANOTHER-INDEX-NEXT:     <div>
// HTML-ANOTHER-INDEX-NEXT:       <h3 id="{{([0-9A-F]{40})}}">functionInAnotherNamespace</h3>
// HTML-ANOTHER-INDEX-NEXT:       <p>void functionInAnotherNamespace()</p>
// HTML-ANOTHER-INDEX-NEXT:       <p>Defined at line 54 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-ANOTHER-INDEX-NEXT:       <div>
// HTML-ANOTHER-INDEX-NEXT:         <div>
// HTML-ANOTHER-INDEX-NEXT:           <p> Function in AnotherNamespace</p>
// HTML-ANOTHER-INDEX-NEXT:         </div>
// HTML-ANOTHER-INDEX-NEXT:       </div>
// HTML-ANOTHER-INDEX-NEXT:     </div>
// HTML-ANOTHER-INDEX-NEXT:   </div>
// HTML-ANOTHER-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-ANOTHER-INDEX-NEXT:     <ol>
// HTML-ANOTHER-INDEX-NEXT:       <li>
// HTML-ANOTHER-INDEX-NEXT:         <span>
// HTML-ANOTHER-INDEX-NEXT:           <a href="#Records">Records</a>
// HTML-ANOTHER-INDEX-NEXT:         </span>
// HTML-ANOTHER-INDEX-NEXT:       </li>
// HTML-ANOTHER-INDEX-NEXT:       <li>
// HTML-ANOTHER-INDEX-NEXT:         <span>
// HTML-ANOTHER-INDEX-NEXT:           <a href="#Functions">Functions</a>
// HTML-ANOTHER-INDEX-NEXT:         </span>
// HTML-ANOTHER-INDEX-NEXT:         <ul>
// HTML-ANOTHER-INDEX-NEXT:           <li>
// HTML-ANOTHER-INDEX-NEXT:             <span>
// HTML-ANOTHER-INDEX-NEXT:               <a href="#{{([0-9A-F]{40})}}">functionInAnotherNamespace</a>
// HTML-ANOTHER-INDEX-NEXT:             </span>
// HTML-ANOTHER-INDEX-NEXT:           </li>
// HTML-ANOTHER-INDEX-NEXT:         </ul>
// HTML-ANOTHER-INDEX-NEXT:       </li>
// HTML-ANOTHER-INDEX-NEXT:     </ol>
// HTML-ANOTHER-INDEX-NEXT:   </div>
// HTML-ANOTHER-INDEX-NEXT: </main>

// HTML-GLOBAL-INDEX: <!DOCTYPE html>
// HTML-GLOBAL-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-GLOBAL-INDEX-NEXT: <title>Global Namespace</title>
// HTML-GLOBAL-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-GLOBAL-INDEX-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-GLOBAL-INDEX-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-GLOBAL-INDEX-NEXT: <header id="project-title"></header>
// HTML-GLOBAL-INDEX-NEXT: <main>
// HTML-GLOBAL-INDEX-NEXT:   <div id="sidebar-left" path="GlobalNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-GLOBAL-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-GLOBAL-INDEX-NEXT:     <h1>Global Namespace</h1>
// HTML-GLOBAL-INDEX-NEXT:     <h2 id="Namespaces">Namespaces</h2>
// HTML-GLOBAL-INDEX-NEXT:     <ul>
// HTML-GLOBAL-INDEX-NEXT:       <li>@nonymous_namespace</li>
// HTML-GLOBAL-INDEX-NEXT:       <li>PrimaryNamespace</li>
// HTML-GLOBAL-INDEX-NEXT:       <li>AnotherNamespace</li>
// HTML-GLOBAL-INDEX-NEXT:     </ul>
// HTML-GLOBAL-INDEX-NEXT:   </div>
// HTML-GLOBAL-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-GLOBAL-INDEX-NEXT:     <ol>
// HTML-GLOBAL-INDEX-NEXT:       <li>
// HTML-GLOBAL-INDEX-NEXT:         <span>
// HTML-GLOBAL-INDEX-NEXT:           <a href="#Namespaces">Namespaces</a>
// HTML-GLOBAL-INDEX-NEXT:         </span>
// HTML-GLOBAL-INDEX-NEXT:       </li>
// HTML-GLOBAL-INDEX-NEXT:     </ol>
// HTML-GLOBAL-INDEX-NEXT:   </div>
// HTML-GLOBAL-INDEX-NEXT: </main>

// HTML-NESTED-CLASS: <!DOCTYPE html>
// HTML-NESTED-CLASS-NEXT: <meta charset="utf-8"/>
// HTML-NESTED-CLASS-NEXT: <title>class ClassInNestedNamespace</title>
// HTML-NESTED-CLASS-NEXT: <link rel="stylesheet" href="..{{[\/]}}..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-NESTED-CLASS-NEXT: <script src="..{{[\/]}}..{{[\/]}}index_json.js"></script>
// HTML-NESTED-CLASS-NEXT: <script src="..{{[\/]}}..{{[\/]}}index.js"></script>
// HTML-NESTED-CLASS-NEXT: <header id="project-title"></header>
// HTML-NESTED-CLASS-NEXT: <main>
// HTML-NESTED-CLASS-NEXT:   <div id="sidebar-left" path="PrimaryNamespace\NestedNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-NESTED-CLASS-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-NESTED-CLASS-NEXT:     <h1>class ClassInNestedNamespace</h1>
// HTML-NESTED-CLASS-NEXT:     <p>Defined at line 47 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-NESTED-CLASS-NEXT:     <div>
// HTML-NESTED-CLASS-NEXT:       <div>
// HTML-NESTED-CLASS-NEXT:         <p> Class in NestedNamespace</p>
// HTML-NESTED-CLASS-NEXT:       </div>
// HTML-NESTED-CLASS-NEXT:     </div>
// HTML-NESTED-CLASS-NEXT:   </div>
// HTML-NESTED-CLASS-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right"></div>
// HTML-NESTED-CLASS-NEXT: </main>

// HTML-NESTED-INDEX: <!DOCTYPE html>
// HTML-NESTED-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-NESTED-INDEX-NEXT: <title>namespace NestedNamespace</title>
// HTML-NESTED-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-NESTED-INDEX-NEXT: <script src="..{{[\/]}}..{{[\/]}}index_json.js"></script>
// HTML-NESTED-INDEX-NEXT: <script src="..{{[\/]}}..{{[\/]}}index.js"></script>
// HTML-NESTED-INDEX-NEXT: <header id="project-title"></header>
// HTML-NESTED-INDEX-NEXT: <main>
// HTML-NESTED-INDEX-NEXT:   <div id="sidebar-left" path="PrimaryNamespace{{[\/]}}NestedNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-NESTED-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-NESTED-INDEX-NEXT:     <h1>namespace NestedNamespace</h1>
// HTML-NESTED-INDEX-NEXT:     <div>
// HTML-NESTED-INDEX-NEXT:       <div>
// HTML-NESTED-INDEX-NEXT:         <p> Nested namespace</p>
// HTML-NESTED-INDEX-NEXT:       </div>
// HTML-NESTED-INDEX-NEXT:     </div>
// HTML-NESTED-INDEX-NEXT:     <h2 id="Records">Records</h2>
// HTML-NESTED-INDEX-NEXT:     <ul>
// HTML-NESTED-INDEX-NEXT:       <li>
// HTML-NESTED-INDEX-NEXT:         <a href="ClassInNestedNamespace.html">ClassInNestedNamespace</a>
// HTML-NESTED-INDEX-NEXT:       </li>
// HTML-NESTED-INDEX-NEXT:     </ul>
// HTML-NESTED-INDEX-NEXT:     <h2 id="Functions">Functions</h2>
// HTML-NESTED-INDEX-NEXT:     <div>
// HTML-NESTED-INDEX-NEXT:       <h3 id="{{([0-9A-F]{40})}}">functionInNestedNamespace</h3>
// HTML-NESTED-INDEX-NEXT:       <p>void functionInNestedNamespace()</p>
// HTML-NESTED-INDEX-NEXT:       <p>Defined at line 45 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-NESTED-INDEX-NEXT:       <div>
// HTML-NESTED-INDEX-NEXT:         <div>
// HTML-NESTED-INDEX-NEXT:           <p> Function in NestedNamespace</p>
// HTML-NESTED-INDEX-NEXT:         </div>
// HTML-NESTED-INDEX-NEXT:       </div>
// HTML-NESTED-INDEX-NEXT:     </div>
// HTML-NESTED-INDEX-NEXT:   </div>
// HTML-NESTED-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-NESTED-INDEX-NEXT:     <ol>
// HTML-NESTED-INDEX-NEXT:       <li>
// HTML-NESTED-INDEX-NEXT:         <span>
// HTML-NESTED-INDEX-NEXT:           <a href="#Records">Records</a>
// HTML-NESTED-INDEX-NEXT:         </span>
// HTML-NESTED-INDEX-NEXT:       </li>
// HTML-NESTED-INDEX-NEXT:       <li>
// HTML-NESTED-INDEX-NEXT:         <span>
// HTML-NESTED-INDEX-NEXT:           <a href="#Functions">Functions</a>
// HTML-NESTED-INDEX-NEXT:         </span>
// HTML-NESTED-INDEX-NEXT:         <ul>
// HTML-NESTED-INDEX-NEXT:           <li>
// HTML-NESTED-INDEX-NEXT:             <span>
// HTML-NESTED-INDEX-NEXT:               <a href="#{{([0-9A-F]{40})}}">functionInNestedNamespace</a>
// HTML-NESTED-INDEX-NEXT:             </span>
// HTML-NESTED-INDEX-NEXT:           </li>
// HTML-NESTED-INDEX-NEXT:         </ul>
// HTML-NESTED-INDEX-NEXT:       </li>
// HTML-NESTED-INDEX-NEXT:     </ol>
// HTML-NESTED-INDEX-NEXT:   </div>
// HTML-NESTED-INDEX-NEXT: </main>

// HTML-PRIMARY-CLASS: <!DOCTYPE html>
// HTML-PRIMARY-CLASS-NEXT: <meta charset="utf-8"/>
// HTML-PRIMARY-CLASS-NEXT: <title>class ClassInPrimaryNamespace</title>
// HTML-PRIMARY-CLASS-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-PRIMARY-CLASS-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-PRIMARY-CLASS-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-PRIMARY-CLASS-NEXT: <header id="project-title"></header>
// HTML-PRIMARY-CLASS-NEXT: <main>
// HTML-PRIMARY-CLASS-NEXT:   <div id="sidebar-left" path="PrimaryNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-PRIMARY-CLASS-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-PRIMARY-CLASS-NEXT:     <h1>class ClassInPrimaryNamespace</h1>
// HTML-PRIMARY-CLASS-NEXT:     <p>Defined at line 40 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-PRIMARY-CLASS-NEXT:     <div>
// HTML-PRIMARY-CLASS-NEXT:       <div>
// HTML-PRIMARY-CLASS-NEXT:         <p> Class in PrimaryNamespace</p>
// HTML-PRIMARY-CLASS-NEXT:       </div>
// HTML-PRIMARY-CLASS-NEXT:     </div>
// HTML-PRIMARY-CLASS-NEXT:   </div>
// HTML-PRIMARY-CLASS-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right"></div>
// HTML-PRIMARY-CLASS-NEXT: </main>

// HTML-PRIMARY-INDEX: <!DOCTYPE html>
// HTML-PRIMARY-INDEX-NEXT: <meta charset="utf-8"/>
// HTML-PRIMARY-INDEX-NEXT: <title>namespace PrimaryNamespace</title>
// HTML-PRIMARY-INDEX-NEXT: <link rel="stylesheet" href="..{{[\/]}}clang-doc-default-stylesheet.css"/>
// HTML-PRIMARY-INDEX-NEXT: <script src="..{{[\/]}}index_json.js"></script>
// HTML-PRIMARY-INDEX-NEXT: <script src="..{{[\/]}}index.js"></script>
// HTML-PRIMARY-INDEX-NEXT: <header id="project-title"></header>
// HTML-PRIMARY-INDEX-NEXT: <main>
// HTML-PRIMARY-INDEX-NEXT:   <div id="sidebar-left" path="PrimaryNamespace" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left"></div>
// HTML-PRIMARY-INDEX-NEXT:   <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-PRIMARY-INDEX-NEXT:     <h1>namespace PrimaryNamespace</h1>
// HTML-PRIMARY-INDEX-NEXT:     <div>
// HTML-PRIMARY-INDEX-NEXT:       <div>
// HTML-PRIMARY-INDEX-NEXT:         <p> Primary Namespace</p>
// HTML-PRIMARY-INDEX-NEXT:       </div>
// HTML-PRIMARY-INDEX-NEXT:     </div>
// HTML-PRIMARY-INDEX-NEXT:     <h2 id="Namespaces">Namespaces</h2>
// HTML-PRIMARY-INDEX-NEXT:     <ul>
// HTML-PRIMARY-INDEX-NEXT:       <li>
// HTML-PRIMARY-INDEX-NEXT:         <a href="NestedNamespace{{[\/]}}index.html">NestedNamespace</a>
// HTML-PRIMARY-INDEX-NEXT:       </li>
// HTML-PRIMARY-INDEX-NEXT:     </ul>
// HTML-PRIMARY-INDEX-NEXT:     <h2 id="Records">Records</h2>
// HTML-PRIMARY-INDEX-NEXT:     <ul>
// HTML-PRIMARY-INDEX-NEXT:       <li>
// HTML-PRIMARY-INDEX-NEXT:         <a href="ClassInPrimaryNamespace.html">ClassInPrimaryNamespace</a>
// HTML-PRIMARY-INDEX-NEXT:       </li>
// HTML-PRIMARY-INDEX-NEXT:     </ul>
// HTML-PRIMARY-INDEX-NEXT:     <h2 id="Functions">Functions</h2>
// HTML-PRIMARY-INDEX-NEXT:     <div>
// HTML-PRIMARY-INDEX-NEXT:       <h3 id="{{([0-9A-F]{40})}}">functionInPrimaryNamespace</h3>
// HTML-PRIMARY-INDEX-NEXT:       <p>void functionInPrimaryNamespace()</p>
// HTML-PRIMARY-INDEX-NEXT:       <p>Defined at line 37 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-PRIMARY-INDEX-NEXT:       <div>
// HTML-PRIMARY-INDEX-NEXT:         <div>
// HTML-PRIMARY-INDEX-NEXT:           <p> Function in PrimaryNamespace</p>
// HTML-PRIMARY-INDEX-NEXT:         </div>
// HTML-PRIMARY-INDEX-NEXT:       </div>
// HTML-PRIMARY-INDEX-NEXT:     </div>
// HTML-PRIMARY-INDEX-NEXT:   </div>
// HTML-PRIMARY-INDEX-NEXT:   <div id="sidebar-right" class="col-xs-6 col-sm-6 col-md-2 sidebar sidebar-offcanvas-right">
// HTML-PRIMARY-INDEX-NEXT:     <ol>
// HTML-PRIMARY-INDEX-NEXT:       <li>
// HTML-PRIMARY-INDEX-NEXT:         <span>
// HTML-PRIMARY-INDEX-NEXT:           <a href="#Namespaces">Namespaces</a>
// HTML-PRIMARY-INDEX-NEXT:         </span>
// HTML-PRIMARY-INDEX-NEXT:       </li>
// HTML-PRIMARY-INDEX-NEXT:       <li>
// HTML-PRIMARY-INDEX-NEXT:         <span>
// HTML-PRIMARY-INDEX-NEXT:           <a href="#Records">Records</a>
// HTML-PRIMARY-INDEX-NEXT:         </span>
// HTML-PRIMARY-INDEX-NEXT:       </li>
// HTML-PRIMARY-INDEX-NEXT:       <li>
// HTML-PRIMARY-INDEX-NEXT:         <span>
// HTML-PRIMARY-INDEX-NEXT:           <a href="#Functions">Functions</a>
// HTML-PRIMARY-INDEX-NEXT:         </span>
// HTML-PRIMARY-INDEX-NEXT:         <ul>
// HTML-PRIMARY-INDEX-NEXT:           <li>
// HTML-PRIMARY-INDEX-NEXT:             <span>
// HTML-PRIMARY-INDEX-NEXT:               <a href="#{{([0-9A-F]{40})}}">functionInPrimaryNamespace</a>
// HTML-PRIMARY-INDEX-NEXT:             </span>
// HTML-PRIMARY-INDEX-NEXT:           </li>
// HTML-PRIMARY-INDEX-NEXT:         </ul>
// HTML-PRIMARY-INDEX-NEXT:       </li>
// HTML-PRIMARY-INDEX-NEXT:     </ol>
// HTML-PRIMARY-INDEX-NEXT:   </div>
// HTML-PRIMARY-INDEX-NEXT: </main>

// MD-ANON-CLASS: # class AnonClass
// MD-ANON-CLASS: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#31*

// MD-ANON-INDEX: # namespace @nonymous_namespace
// MD-ANON-INDEX:  Anonymous Namespace
// MD-ANON-INDEX: ## Records
// MD-ANON-INDEX: * [AnonClass](AnonClass.md)
// MD-ANON-INDEX: ## Functions
// MD-ANON-INDEX: ### anonFunction
// MD-ANON-INDEX: *void anonFunction()*
// MD-ANON-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#30*

// MD-ANOTHER-CLASS: # class ClassInAnotherNamespace
// MD-ANOTHER-CLASS: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#56*
// MD-ANOTHER-CLASS:  Class in AnotherNamespace

// MD-ANOTHER-INDEX: # namespace AnotherNamespace
// MD-ANOTHER-INDEX: AnotherNamespace
// MD-ANOTHER-INDEX: ## Records
// MD-ANOTHER-INDEX: * [ClassInAnotherNamespace](ClassInAnotherNamespace.md)
// MD-ANOTHER-INDEX: ## Functions
// MD-ANOTHER-INDEX: ### functionInAnotherNamespace
// MD-ANOTHER-INDEX: *void functionInAnotherNamespace()*
// MD-ANOTHER-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#54*
// MD-ANOTHER-INDEX: Function in AnotherNamespace

// MD-GLOBAL-INDEX: # Global Namespace
// MD-GLOBAL-INDEX: ## Namespaces
// MD-GLOBAL-INDEX: * [@nonymous_namespace](..{{[\/]}}@nonymous_namespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [PrimaryNamespace](..{{[\/]}}PrimaryNamespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [AnotherNamespace](..{{[\/]}}AnotherNamespace{{[\/]}}index.md)

// MD-NESTED-CLASS: # class ClassInNestedNamespace
// MD-NESTED-CLASS: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#47*
// MD-NESTED-CLASS: Class in NestedNamespace

// MD-NESTED-INDEX: # namespace NestedNamespace
// MD-NESTED-INDEX: Nested namespace
// MD-NESTED-INDEX: ## Records
// MD-NESTED-INDEX: * [ClassInNestedNamespace](ClassInNestedNamespace.md)
// MD-NESTED-INDEX: ## Functions
// MD-NESTED-INDEX: ### functionInNestedNamespace
// MD-NESTED-INDEX: *void functionInNestedNamespace()*
// MD-NESTED-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#45*
// MD-NESTED-INDEX: Function in NestedNamespace

// MD-PRIMARY-CLASS: # class ClassInPrimaryNamespace
// MD-PRIMARY-CLASS: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#40*
// MD-PRIMARY-CLASS: Class in PrimaryNamespace

// MD-PRIMARY-INDEX: # namespace PrimaryNamespace
// MD-PRIMARY-INDEX:  Primary Namespace
// MD-PRIMARY-INDEX: ## Namespaces
// MD-PRIMARY-INDEX: * [NestedNamespace](NestedNamespace{{[\/]}}index.md)
// MD-PRIMARY-INDEX: ## Records
// MD-PRIMARY-INDEX: * [ClassInPrimaryNamespace](ClassInPrimaryNamespace.md)
// MD-PRIMARY-INDEX: ## Functions
// MD-PRIMARY-INDEX: ### functionInPrimaryNamespace
// MD-PRIMARY-INDEX: *void functionInPrimaryNamespace()*
// MD-PRIMARY-INDEX: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#37*
// MD-PRIMARY-INDEX:  Function in PrimaryNamespace

// MD-ALL-FILES: # All Files
// MD-ALL-FILES: ## [@nonymous_namespace](@nonymous_namespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [AnotherNamespace](AnotherNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [GlobalNamespace](GlobalNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [PrimaryNamespace](PrimaryNamespace{{[\/]}}index.md)

// MD-INDEX: #  C/C++ Reference
// MD-INDEX: * Namespace: [@nonymous_namespace](@nonymous_namespace)
// MD-INDEX: * Namespace: [AnotherNamespace](AnotherNamespace)
// MD-INDEX: * Namespace: [PrimaryNamespace](PrimaryNamespace)