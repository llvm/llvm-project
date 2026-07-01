// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --doxygen --format=html --executor=standalone %S/../Inputs/compound-constraints.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html -check-prefix=CHECK-HTML

// CHECK-HTML:          <a class="sidebar-item" href="#Concepts">Concepts</a>
// CHECK-HTML-NEXT: </summary>
// CHECK-HTML-NEXT: <ul>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Incrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Decrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">PreIncrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT:     <li class="sidebar-item-container">
// CHECK-HTML-NEXT:         <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">PreDecrementable</a>
// CHECK-HTML-NEXT:     </li>
// CHECK-HTML-NEXT: </ul>
// CHECK-HTML:      <section id="Concepts" class="section-container">
// CHECK-HTML-NEXT:     <h2>Concepts</h2>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; Incrementable requires (T a) { a++; }</code></pre>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p>Concept for an incrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be incremented.
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <p>Defined at line 4 of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; Decrementable requires (T a) { a--; }</code></pre>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p>Concept for a decrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be decremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <p>Defined at line 10 of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; PreIncrementable requires (T a) { ++a; }</code></pre>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p>Concept for a pre-incrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be pre-incremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <p>Defined at line 16 of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML-NEXT:         <div>
// CHECK-HTML-NEXT:             <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt; PreDecrementable requires (T a) { --a; }</code></pre>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <p>Concept for a -pre-decrementable value</p>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <h3>Template Parameters</h3>
// CHECK-HTML-NEXT:         <div class="nested-delimiter-container">
// CHECK-HTML-NEXT:             <div>
// CHECK-HTML-NEXT:                 <b>T</b>   A value that can be pre-decremented
// CHECK-HTML-NEXT:             </div>
// CHECK-HTML-NEXT:         </div>
// CHECK-HTML-NEXT:         <p>Defined at line 22 of file {{.*}}compound-constraints.cpp</p>
// CHECK-HTML-NEXT:     </div>
// CHECK-HTML-NEXT: </section>
