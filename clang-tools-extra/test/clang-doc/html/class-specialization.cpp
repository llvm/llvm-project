// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/../Inputs/class-specialization.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=HTML-NAMESPACE

// HTML-NAMESPACE:      <section id="Records" class="section-container">
// HTML-NAMESPACE-NEXT:     <h2>Records</h2>
// HTML-NAMESPACE-NEXT:     <ul class="class-container">
// HTML-NAMESPACE-NEXT:         <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-NAMESPACE-NEXT:             <a href="_ZTV7MyClass.html">
// HTML-NAMESPACE-NEXT:                 <pre><code class="language-cpp code-clang-doc">class MyClass</code></pre>
// HTML-NAMESPACE-NEXT:             </a>
// HTML-NAMESPACE-NEXT:         </li>
// HTML-NAMESPACE-NEXT:         <li id="{{([0-9A-F]{40})}}" style="max-height: 40px;">
// HTML-NAMESPACE-NEXT:             <a href="_ZTV7MyClassIiE.html">
// HTML-NAMESPACE-NEXT:                 <pre><code class="language-cpp code-clang-doc">class MyClass&lt;int&gt;</code></pre>
// HTML-NAMESPACE-NEXT:             </a>
// HTML-NAMESPACE-NEXT:         </li>
// HTML-NAMESPACE-NEXT:     </ul>
// HTML-NAMESPACE-NEXT: </section>
