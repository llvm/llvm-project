// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/../Inputs/class.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html -check-prefix=HTML

// HTML:              <a class="sidebar-item" href="#Records">Records</a>
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
// HTML:      <section id="ProtectedMethods" class="section-container">
// HTML-NEXT:     <h2>Protected Methods</h2>
// HTML-NEXT:     <div>
// HTML-NEXT:         <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-NEXT:                 <pre><code class="language-cpp code-clang-doc">int protectedMethod ()</code></pre>
// HTML-NEXT:         </div>
// HTML-NEXT:     </div>
// HTML-NEXT: </section>
// HTML:      <section id="Records" class="section-container">
// HTML-NEXT:     <h2>Records</h2>
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
// HTML-NEXT:     </div>
// HTML-NEXT:     <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// HTML-NEXT:         <pre><code class="language-cpp code-clang-doc">class Foo</code></pre>
// HTML-NEXT:     </div>
// HTML-NEXT: </section>
