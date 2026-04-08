// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/MyNamespace/index.html
// RUN: FileCheck %s < %t/html/GlobalNamespace/index.html --check-prefix=CHECK-GLOBAL

namespace MyNamespace {
  class Foo;
}

// CHECK:       <ul class="class-container">
// CHECK-NEXT:    <li id="{{[0-9A-F]*}}" style="max-height: 40px;">
// CHECK-NEXT:        <a href="_ZTVN11MyNamespace3FooE.html">
// CHECK-NEXT:            <pre><code class="language-cpp code-clang-doc">class Foo</code></pre>
// CHECK-NEXT:        </a>
// CHECK-NEXT:    </li>
// CHECK-NEXT: </ul>

// COM: Check that the empty global namespace doesn't contain tag mismatches.
// CHECK-GLOBAL:             <main>
// CHECK-GLOBAL-NEXT:            <div class="container">
// CHECK-GLOBAL-NEXT:                <div class="sidebar">
// CHECK-GLOBAL-NEXT:                    <h2>Global Namespace</h2>
// CHECK-GLOBAL-NEXT:                    <ul>
// CHECK-GLOBAL-NEXT:                        <li>
// CHECK-GLOBAL-NEXT:                            <details open>
// CHECK-GLOBAL-NEXT:                                <summary class="sidebar-section">
// CHECK-GLOBAL-NEXT:                                    <a class="sidebar-item" href="#Namespaces">Namespaces</a>
// CHECK-GLOBAL-NEXT:                                </summary>
// CHECK-GLOBAL-NEXT:                                <ul>
// CHECK-GLOBAL-NEXT:                                    <li class="sidebar-item-container">
// CHECK-GLOBAL-NEXT:                                        <a class="sidebar-item" href="#{{[0-9A-F]*}}">MyNamespace</a>
// CHECK-GLOBAL-NEXT:                                    </li>
// CHECK-GLOBAL-NEXT:                                </ul>
// CHECK-GLOBAL-NEXT:                            </details>
// CHECK-GLOBAL-NEXT:                        </li>
// CHECK-GLOBAL-NEXT:                    </ul>
// CHECK-GLOBAL-NEXT:                </div>
// CHECK-GLOBAL-NEXT:                <div class="resizer" id="resizer"></div>
// CHECK-GLOBAL-NEXT:                <div class="content">
// CHECK-GLOBAL-NEXT:                    <section id="Namespaces" class="section-container">
// CHECK-GLOBAL-NEXT:                        <h2>Namespaces</h2>
// CHECK-GLOBAL-NEXT:                        <ul class="class-container">
// CHECK-GLOBAL-NEXT:                            <li id="{{[0-9A-F]*}}">
// CHECK-GLOBAL-NEXT:                                <a href="../MyNamespace/index.html">
// CHECK-GLOBAL-NEXT:                                    <pre><code class="language-cpp code-clang-doc">namespace MyNamespace</code></pre>
// CHECK-GLOBAL-NEXT:                                </a>
// CHECK-GLOBAL-NEXT:                            </li>
// CHECK-GLOBAL-NEXT:                        </ul>
// CHECK-GLOBAL-NEXT:                    </section>
// CHECK-GLOBAL-NEXT:                </div>
// CHECK-GLOBAL-NEXT:            </div>
// CHECK-GLOBAL-NEXT:        </main>
