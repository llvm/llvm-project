// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --output=%t --executor=standalone %s
// RUN: clang-doc --format=md --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/@nonymous_namespace/AnonClass.md -check-prefix=MD-ANON-CLASS-LINE
// RUN: FileCheck %s < %t/@nonymous_namespace/AnonClass.md -check-prefix=MD-ANON-CLASS
// RUN: FileCheck %s < %t/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX-LINE
// RUN: FileCheck %s < %t/@nonymous_namespace/index.md -check-prefix=MD-ANON-INDEX
// RUN: FileCheck %s < %t/AnotherNamespace/ClassInAnotherNamespace.md -check-prefix=MD-ANOTHER-CLASS-LINE
// RUN: FileCheck %s < %t/AnotherNamespace/ClassInAnotherNamespace.md -check-prefix=MD-ANOTHER-CLASS
// RUN: FileCheck %s < %t/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX-LINE
// RUN: FileCheck %s < %t/AnotherNamespace/index.md -check-prefix=MD-ANOTHER-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.md -check-prefix=MD-NESTED-CLASS-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/ClassInNestedNamespace.md -check-prefix=MD-NESTED-CLASS
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/NestedNamespace/index.md -check-prefix=MD-NESTED-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/index.md -check-prefix=MD-PRIMARY-INDEX
// RUN: FileCheck %s < %t/PrimaryNamespace/ClassInPrimaryNamespace.md -check-prefix=MD-PRIMARY-CLASS-LINE
// RUN: FileCheck %s < %t/PrimaryNamespace/ClassInPrimaryNamespace.md -check-prefix=MD-PRIMARY-CLASS
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html -check-prefix=HTML-GLOBAL-INDEX
// RUN: FileCheck %s < %t/GlobalNamespace/index.md -check-prefix=MD-GLOBAL-INDEX
// RUN: FileCheck %s < %t/all_files.md -check-prefix=MD-ALL-FILES
// RUN: FileCheck %s < %t/index.md -check-prefix=MD-INDEX
// RUN: FileCheck %s < %t/html/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.html -check-prefix=HTML-ANON-CLASS-LINE
// RUN: FileCheck %s < %t/html/@nonymous_namespace/_ZTVN12_GLOBAL__N_19AnonClassE.html -check-prefix=HTML-ANON-CLASS
// RUN: FileCheck %s < %t/html/@nonymous_namespace/index.html -check-prefix=HTML-ANON-INDEX-LINE
// RUN: FileCheck %s < %t/html/@nonymous_namespace/index.html -check-prefix=HTML-ANON-INDEX
// RUN: FileCheck %s < %t/html/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.html -check-prefix=HTML-ANOTHER-CLASS-LINE
// RUN: FileCheck %s < %t/html/AnotherNamespace/_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.html -check-prefix=HTML-ANOTHER-CLASS
// RUN: FileCheck %s < %t/html/AnotherNamespace/index.html -check-prefix=HTML-ANOTHER-INDEX-LINE
// RUN: FileCheck %s < %t/html/AnotherNamespace/index.html -check-prefix=HTML-ANOTHER-INDEX
// RUN: FileCheck %s < %t/html/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.html -check-prefix=HTML-NESTED-CLASS-LINE
// RUN: FileCheck %s < %t/html/PrimaryNamespace/NestedNamespace/_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.html -check-prefix=HTML-NESTED-CLASS
// RUN: FileCheck %s < %t/html/PrimaryNamespace/NestedNamespace/index.html -check-prefix=HTML-NESTED-INDEX-LINE
// RUN: FileCheck %s < %t/html/PrimaryNamespace/NestedNamespace/index.html -check-prefix=HTML-NESTED-INDEX
// RUN: FileCheck %s < %t/html/PrimaryNamespace/index.html -check-prefix=HTML-PRIMARY-INDEX-LINE
// RUN: FileCheck %s < %t/html/PrimaryNamespace/index.html -check-prefix=HTML-PRIMARY-INDEX
// RUN: FileCheck %s < %t/html/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.html -check-prefix=HTML-PRIMARY-CLASS-LINE
// RUN: FileCheck %s < %t/html/PrimaryNamespace/_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.html -check-prefix=HTML-PRIMARY-CLASS

// COM: FIXME: Add global functions to the namespace template
// COM: FIXME: Add namespaces to the namespace template

// Anonymous Namespace
namespace {
void anonFunction() {}
// MD-ANON-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-ANON-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

class AnonClass {};
// MD-ANON-CLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-ANON-CLASS-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// MD-ANON-CLASS: # class AnonClass
// HTML-ANON-CLASS: <h1 class="hero__title-large">class AnonClass</h1>
} // namespace

// MD-ANON-INDEX: # namespace @nonymous_namespace
// MD-ANON-INDEX:  Anonymous Namespace
// MD-ANON-INDEX: ## Records
// MD-ANON-INDEX: * [AnonClass](AnonClass.md)
// MD-ANON-INDEX: ## Functions
// MD-ANON-INDEX: ### anonFunction
// MD-ANON-INDEX: *void anonFunction()*

// HTML-ANON-INDEX: <div class="navbar-breadcrumb-container">
// HTML-ANON-INDEX:     <a href="../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>
// HTML-ANON-INDEX: </div>
// HTML-ANON-INDEX: <h2>@nonymous_namespace</h2>
// HTML-ANON-INDEX:     <h2>Inner Classes</h2>
// HTML-ANON-INDEX:         <ul class="class-container">
// HTML-ANON-INDEX:             <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-ANON-INDEX:                 <a href="_ZTVN12_GLOBAL__N_19AnonClassE.html">
// HTML-ANON-INDEX:                     <pre><code class="language-cpp code-clang-doc">class AnonClass</code></pre>
// HTML-ANON-INDEX:                 </a>
// HTML-ANON-INDEX:             </li>
// HTML-ANON-INDEX-NOT: <h2 id="Functions">Functions</h2>
// HTML-ANON-INDEX-NOT: <h3 id="{{([0-9A-F]{40})}}">anonFunction</h3>
// HTML-ANON-INDEX-NOT: <p>void anonFunction()</p>

// Primary Namespace
namespace PrimaryNamespace {
// Function in PrimaryNamespace
void functionInPrimaryNamespace() {}
// MD-PRIMARY-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-PRIMARY-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// Class in PrimaryNamespace
class ClassInPrimaryNamespace {};
// MD-PRIMARY-CLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-PRIMARY-CLASS-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// MD-PRIMARY-CLASS: # class ClassInPrimaryNamespace
// MD-PRIMARY-CLASS: Class in PrimaryNamespace

// HTML-PRIMARY-CLASS: <div class="navbar-breadcrumb-container">
// HTML-PRIMARY-CLASS:     <a href="../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>::
// HTML-PRIMARY-CLASS:     <a href="./index.html"><div class="navbar-breadcrumb-item">PrimaryNamespace</div></a>
// HTML-PRIMARY-CLASS: </div>
// HTML-PRIMARY-CLASS: <h1 class="hero__title-large">class ClassInPrimaryNamespace</h1>

// Nested namespace
namespace NestedNamespace {
// Function in NestedNamespace
void functionInNestedNamespace() {}
// MD-NESTED-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-NESTED-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// Class in NestedNamespace
class ClassInNestedNamespace {};
// MD-NESTED-CLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-NESTED-CLASS-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// MD-NESTED-CLASS: # class ClassInNestedNamespace
// MD-NESTED-CLASS: Class in NestedNamespace

// HTML-NESTED-CLASS: <div class="navbar-breadcrumb-container">
// HTML-NESTED-CLASS:     <a href="../../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>::
// HTML-NESTED-CLASS:     <a href="../index.html"><div class="navbar-breadcrumb-item">PrimaryNamespace</div></a>::
// HTML-NESTED-CLASS:     <a href="./index.html"><div class="navbar-breadcrumb-item">NestedNamespace</div></a>
// HTML-NESTED-CLASS: </div>
// HTML-NESTED-CLASS: <h1 class="hero__title-large">class ClassInNestedNamespace</h1>
} // namespace NestedNamespace

// MD-NESTED-INDEX: # namespace NestedNamespace
// MD-NESTED-INDEX: Nested namespace
// MD-NESTED-INDEX: ## Records
// MD-NESTED-INDEX: * [ClassInNestedNamespace](ClassInNestedNamespace.md)
// MD-NESTED-INDEX: ## Functions
// MD-NESTED-INDEX: ### functionInNestedNamespace
// MD-NESTED-INDEX: *void functionInNestedNamespace()*
// MD-NESTED-INDEX: Function in NestedNamespace

// HTML-NESTED-INDEX: <div class="navbar-breadcrumb-container">
// HTML-NESTED-INDEX:     <a href="../../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>::
// HTML-NESTED-INDEX:     <a href="../index.html"><div class="navbar-breadcrumb-item">PrimaryNamespace</div></a>
// HTML-NESTED-INDEX: </div>
// HTML-NESTED-INDEX: <h2>NestedNamespace</h2>
// HTML-NESTED-INDEX:     <h2>Inner Classes</h2>
// HTML-NESTED-INDEX:     <ul class="class-container">
// HTML-NESTED-INDEX:         <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-NESTED-INDEX:             <a href="_ZTVN16PrimaryNamespace15NestedNamespace22ClassInNestedNamespaceE.html">
// HTML-NESTED-INDEX:                 <pre><code class="language-cpp code-clang-doc">class ClassInNestedNamespace</code></pre>
// HTML-NESTED-INDEX:             </a>
// HTML-NESTED-INDEX:         </li>
// HTML-NESTED-INDEX:     </ul>
// HTML-NESTED-INDEX:         <pre><code class="language-cpp code-clang-doc">void functionInNestedNamespace ()</code></pre>
// HTML-NESTED-INDEX:         <div class="doc-card">
// HTML-NESTED-INDEX:             <div class="nested-delimiter-container">
// HTML-NESTED-INDEX:                 <p> Function in NestedNamespace</p>
// HTML-NESTED-INDEX:             </div>
// HTML-NESTED-INDEX:         </div>
// HTML-NESTED-INDEX:         <p>Defined at line 105 of file {{.*}}namespace.cpp</p>
// HTML-NESTED-INDEX:     </div>
} // namespace PrimaryNamespace

// MD-PRIMARY-INDEX: # namespace PrimaryNamespace
// MD-PRIMARY-INDEX:  Primary Namespace
// MD-PRIMARY-INDEX: ## Namespaces
// MD-PRIMARY-INDEX: * [NestedNamespace](NestedNamespace{{[\/]}}index.md)
// MD-PRIMARY-INDEX: ## Records
// MD-PRIMARY-INDEX: * [ClassInPrimaryNamespace](ClassInPrimaryNamespace.md)
// MD-PRIMARY-INDEX: ## Functions
// MD-PRIMARY-INDEX: ### functionInPrimaryNamespace
// MD-PRIMARY-INDEX: *void functionInPrimaryNamespace()*
// MD-PRIMARY-INDEX:  Function in PrimaryNamespace

// HTML-PRIMARY-INDEX: <div class="navbar-breadcrumb-container">
// HTML-PRIMARY-INDEX:     <a href="../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>
// HTML-PRIMARY-INDEX: </div>
// HTML-PRIMARY-INDEX: <h2>PrimaryNamespace</h2>
// HTML-PRIMARY-INDEX-NOT: <h2 id="Namespaces">Namespaces</h2>
// HTML-PRIMARY-INDEX-NOT: <a href="NestedNamespace{{[\/]}}index.html">NestedNamespace</a>
// HTML-PRIMARY-INDEX:      <h2>Inner Classes</h2>
// HTML-PRIMARY-INDEX:          <ul class="class-container">
// HTML-PRIMARY-INDEX:              <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-PRIMARY-INDEX:                  <a href="_ZTVN16PrimaryNamespace23ClassInPrimaryNamespaceE.html">
// HTML-PRIMARY-INDEX:                      <pre><code class="language-cpp code-clang-doc">class ClassInPrimaryNamespace</code></pre>
// HTML-PRIMARY-INDEX:                 </a>
// HTML-PRIMARY-INDEX:              </li>
// HTML-PRIMARY-INDEX:          </ul>
// HTML-PRIMARY-INDEX:          <pre><code class="language-cpp code-clang-doc">void functionInPrimaryNamespace ()</code></pre>
// HTML-PRIMARY-INDEX:          <div class="doc-card">
// HTML-PRIMARY-INDEX:              <div class="nested-delimiter-container">
// HTML-PRIMARY-INDEX:                  <p> Function in PrimaryNamespace</p>
// HTML-PRIMARY-INDEX:              </div>
// HTML-PRIMARY-INDEX:          </div>
// HTML-PRIMARY-INDEX:          <p>Defined at line 84 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-PRIMARY-INDEX:      </div>
// AnotherNamespace
namespace AnotherNamespace {
// Function in AnotherNamespace
void functionInAnotherNamespace() {}
// MD-ANOTHER-INDEX-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-ANOTHER-INDEX-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// Class in AnotherNamespace
class ClassInAnotherNamespace {};
// MD-ANOTHER-CLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp#[[@LINE-1]]*
// HTML-ANOTHER-CLASS-LINE: <p>Defined at line [[@LINE-2]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>

// MD-ANOTHER-CLASS: # class ClassInAnotherNamespace
// MD-ANOTHER-CLASS:  Class in AnotherNamespace

// HTML-ANOTHER-CLASS: <div class="navbar-breadcrumb-container">
// HTML-ANOTHER-CLASS:     <a href="../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>::
// HTML-ANOTHER-CLASS:     <a href="./index.html"><div class="navbar-breadcrumb-item">AnotherNamespace</div></a>
// HTML-ANOTHER-CLASS: </div>
// HTML-ANOTHER-CLASS: <h1 class="hero__title-large">class ClassInAnotherNamespace</h1>

} // namespace AnotherNamespace

// MD-ANOTHER-INDEX: # namespace AnotherNamespace
// MD-ANOTHER-INDEX: AnotherNamespace
// MD-ANOTHER-INDEX: ## Records
// MD-ANOTHER-INDEX: * [ClassInAnotherNamespace](ClassInAnotherNamespace.md)
// MD-ANOTHER-INDEX: ## Functions
// MD-ANOTHER-INDEX: ### functionInAnotherNamespace
// MD-ANOTHER-INDEX: *void functionInAnotherNamespace()*
// MD-ANOTHER-INDEX: Function in AnotherNamespace

// HTML-ANOTHER-INDEX: <div class="navbar-breadcrumb-container">
// HTML-ANOTHER-INDEX:     <a href="../GlobalNamespace/index.html"><div class="navbar-breadcrumb-item">Global Namespace</div></a>
// HTML-ANOTHER-INDEX: </div>
// HTML-ANOTHER-INDEX: <h2>AnotherNamespace</h2>
// HTML-ANOTHER-INDEX:     <h2>Inner Classes</h2>
// HTML-ANOTHER-INDEX:     <ul class="class-container">
// HTML-ANOTHER-INDEX:         <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-ANOTHER-INDEX:             <a href="_ZTVN16AnotherNamespace23ClassInAnotherNamespaceE.html">
// HTML-ANOTHER-INDEX:                 <pre><code class="language-cpp code-clang-doc">class ClassInAnotherNamespace</code></pre>
// HTML-ANOTHER-INDEX:             </a>
// HTML-ANOTHER-INDEX:         </li>
// HTML-ANOTHER-INDEX:     </ul>
// HTML-ANOTHER-INDEX:         <pre><code class="language-cpp code-clang-doc">void functionInAnotherNamespace ()</code></pre>
// HTML-ANOTHER-INDEX:         <div class="doc-card">
// HTML-ANOTHER-INDEX:             <div class="nested-delimiter-container">
// HTML-ANOTHER-INDEX:                 <p> Function in AnotherNamespace</p>
// HTML-ANOTHER-INDEX:             </div>
// HTML-ANOTHER-INDEX:         </div>
// HTML-ANOTHER-INDEX:         <p>Defined at line 193 of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}namespace.cpp</p>
// HTML-ANOTHER-INDEX:     </div>
// HTML-ANOTHER-INDEX: </div>

// COM: FIXME: Add namespaces to namespace template
// HTML-GLOBAL-INDEX-NOT: <div id="main-content" class="col-xs-12 col-sm-9 col-md-8 main-content">
// HTML-GLOBAL-INDEX-NOT: <h1>Global Namespace</h1>
// HTML-GLOBAL-INDEX-NOT: <h2 id="Namespaces">Namespaces</h2>
// HTML-GLOBAL-INDEX-NOT: <li>@nonymous_namespace</li>
// HTML-GLOBAL-INDEX-NOT: <li>AnotherNamespace</li>
// HTML-GLOBAL-INDEX-NOT: <li>PrimaryNamespace</li>

// MD-GLOBAL-INDEX: # Global Namespace
// MD-GLOBAL-INDEX: ## Namespaces
// MD-GLOBAL-INDEX: * [@nonymous_namespace](..{{[\/]}}@nonymous_namespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [AnotherNamespace](..{{[\/]}}AnotherNamespace{{[\/]}}index.md)
// MD-GLOBAL-INDEX: * [PrimaryNamespace](..{{[\/]}}PrimaryNamespace{{[\/]}}index.md)

// MD-ALL-FILES: # All Files
// MD-ALL-FILES: ## [@nonymous_namespace](@nonymous_namespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [AnotherNamespace](AnotherNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [GlobalNamespace](GlobalNamespace{{[\/]}}index.md)
// MD-ALL-FILES: ## [PrimaryNamespace](PrimaryNamespace{{[\/]}}index.md)

// MD-INDEX: #  C/C++ Reference
// MD-INDEX: * Namespace: [@nonymous_namespace](@nonymous_namespace)
// MD-INDEX: * Namespace: [AnotherNamespace](AnotherNamespace)
// MD-INDEX: * Namespace: [PrimaryNamespace](PrimaryNamespace)
