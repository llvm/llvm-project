// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --output=%t --executor=standalone %S/../Inputs/typedef-alias.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html -check-prefix=HTML-GLOBAL
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV6Vector.html -check-prefix=HTML-VECTOR

// HTML-GLOBAL:              <a class="sidebar-item" href="#Typedefs">Typedefs</a>
// HTML-GLOBAL-NEXT:     </summary>
// HTML-GLOBAL-NEXT:     <ul>
// HTML-GLOBAL-NEXT:         <li class="sidebar-item-container">
// HTML-GLOBAL-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">u_long</a>
// HTML-GLOBAL-NEXT:         </li>
// HTML-GLOBAL-NEXT:         <li class="sidebar-item-container">
// HTML-GLOBAL-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">IntPtr</a>
// HTML-GLOBAL-NEXT:         </li>
// HTML-GLOBAL-NEXT:         <li class="sidebar-item-container">
// HTML-GLOBAL-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Vec</a>
// HTML-GLOBAL-NEXT:         </li>
// HTML-GLOBAL-NEXT:         <li class="sidebar-item-container">
// HTML-GLOBAL-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">IntVec</a>
// HTML-GLOBAL-NEXT:         </li>
// HTML-GLOBAL-NEXT:     </ul>
// HTML-GLOBAL:      <section id="Typedefs" class="section-container">
// HTML-GLOBAL-NEXT:     <h2>Typedefs</h2>
// HTML-GLOBAL-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-GLOBAL-NEXT:         <pre><code class="language-cpp code-clang-doc">using u_long = unsigned long</code></pre>
// HTML-GLOBAL-NEXT:         <div class="nested-delimiter-container">
// HTML-GLOBAL-NEXT:             <p>This is u_long</p>
// HTML-GLOBAL-NEXT:         </div>
// HTML-GLOBAL-NEXT:         <p>Defined at line 2 of file {{.*}}typedef-alias.cpp</p>
// HTML-GLOBAL-NEXT:     </div>
// HTML-GLOBAL-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-GLOBAL-NEXT:         <pre><code class="language-cpp code-clang-doc">typedef IntPtr int *</code></pre>
// HTML-GLOBAL-NEXT:         <div class="nested-delimiter-container">
// HTML-GLOBAL-NEXT:             <p>This is IntPtr</p>
// HTML-GLOBAL-NEXT:         </div>
// HTML-GLOBAL-NEXT:         <p>Defined at line 5 of file {{.*}}typedef-alias.cpp</p>
// HTML-GLOBAL-NEXT:     </div>
// HTML-GLOBAL-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-GLOBAL-NEXT:         <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt;</code></pre>
// HTML-GLOBAL-NEXT:         <pre><code class="language-cpp code-clang-doc">using Vec = Vector&lt;T&gt;</code></pre>
// HTML-GLOBAL-NEXT:         <p>Defined at line 12 of file {{.*}}typedef-alias.cpp</p>
// HTML-GLOBAL-NEXT:     </div>
// HTML-GLOBAL-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-GLOBAL-NEXT:         <pre><code class="language-cpp code-clang-doc">using IntVec = Vector&lt;int&gt;</code></pre>
// HTML-GLOBAL-NEXT:         <p>Defined at line 14 of file {{.*}}typedef-alias.cpp</p>
// HTML-GLOBAL-NEXT:     </div>
// HTML-GLOBAL-NEXT: </section>

// HTML-VECTOR:              <a class="sidebar-item" href="#Typedefs">Typedefs</a>
// HTML-VECTOR-NEXT:     </summary>
// HTML-VECTOR-NEXT:     <ul>
// HTML-VECTOR-NEXT:         <li class="sidebar-item-container">
// HTML-VECTOR-NEXT:             <a class="sidebar-item" href="#{{([0-9A-F]{40})}}">Ptr</a>
// HTML-VECTOR-NEXT:         </li>
// HTML-VECTOR-NEXT:     </ul>
// HTML-VECTOR:      <section id="Typedefs" class="section-container">
// HTML-VECTOR-NEXT:     <h2>Typedefs</h2>
// HTML-VECTOR-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-VECTOR-NEXT:         <pre><code class="language-cpp code-clang-doc">using Ptr = IntPtr</code></pre>
// HTML-VECTOR-NEXT:         <div class="nested-delimiter-container">
// HTML-VECTOR-NEXT:             <p>This is a Ptr</p>
// HTML-VECTOR-NEXT:         </div>
// HTML-VECTOR-NEXT:         <p>Defined at line 9 of file {{.*}}typedef-alias.cpp</p>
// HTML-VECTOR-NEXT:     </div>
// HTML-VECTOR-NEXT: </section>
