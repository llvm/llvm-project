// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/json/index.json -check-prefix=CHECK-JSON
// RUN: FileCheck %s < %t/html/index.html -check-prefix=CHECK-HTML

class Foo {};

namespace inner {
  class Bar {};
}

// CHECK-JSON:       "Index": [
// CHECK-JSON-NEXT:    {
// CHECK-JSON-NEXT:      "Name": "GlobalNamespace",
// CHECK-JSON-NEXT:      "QualName": "GlobalNamespace",
// CHECK-JSON-NEXT:      "USR": "0000000000000000000000000000000000000000"
// CHECK-JSON-NEXT:    },
// CHECK-JSON-NEXT:    {
// CHECK-JSON-NEXT:      "Name": "inner",
// CHECK-JSON-NEXT:      "QualName": "inner",
// CHECK-JSON-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// CHECK-JSON-NEXT:    }
// CHECK-JSON-NEXT:  ]

// CHECK-HTML:         <main>
// CHECK-HTML-NEXT:        <div class="container">
// CHECK-HTML-NEXT:            <div class="sidebar">
// CHECK-HTML-NEXT:                <h2></h2>
// CHECK-HTML-NEXT:                <ul>
// CHECK-HTML-NEXT:                    <li>
// CHECK-HTML-NEXT:                        <details open>
// CHECK-HTML-NEXT:                            <summary class="sidebar-section">
// CHECK-HTML-NEXT:                                <a class="sidebar-item" href="#Namespaces">Namespaces</a>
// CHECK-HTML-NEXT:                            </summary>
// CHECK-HTML-NEXT:                            <ul>
// CHECK-HTML-NEXT:                                <li class="sidebar-item-container">
// CHECK-HTML-NEXT:                                    <a class="sidebar-item" href="#GlobalNamespace">GlobalNamespace</a>
// CHECK-HTML-NEXT:                                </li>
// CHECK-HTML-NEXT:                                <li class="sidebar-item-container">
// CHECK-HTML-NEXT:                                    <a class="sidebar-item" href="#inner">inner</a>
// CHECK-HTML-NEXT:                                </li>
// CHECK-HTML-NEXT:                            </ul>
// CHECK-HTML-NEXT:                        </details> 
// CHECK-HTML-NEXT:                    </li>
// CHECK-HTML-NEXT:                </ul>
// CHECK-HTML-NEXT:            </div>
// CHECK-HTML-NEXT:            <div class="resizer" id="resizer"></div>
// CHECK-HTML-NEXT:            <div class="content">
// CHECK-HTML-NEXT:                <section id="Index" class="section-container">
// CHECK-HTML-NEXT:                    <h2>Index</h2>
// CHECK-HTML-NEXT:                    <div>
// CHECK-HTML-NEXT:                        <a href="GlobalNamespace/index.html">
// CHECK-HTML-NEXT:                            <pre><code class="language-cpp code-clang-doc">namespace GlobalNamespace</code></pre>
// CHECK-HTML-NEXT:                        </a>
// CHECK-HTML-NEXT:                    </div>
// CHECK-HTML-NEXT:                    <div>
// CHECK-HTML-NEXT:                        <a href="inner/index.html">
// CHECK-HTML-NEXT:                            <pre><code class="language-cpp code-clang-doc">namespace inner</code></pre>
// CHECK-HTML-NEXT:                        </a>
// CHECK-HTML-NEXT:                    </div>
// CHECK-HTML-NEXT:                </section>
// CHECK-HTML-NEXT:            </div>
// CHECK-HTML-NEXT:        </div>
// CHECK-HTML-NEXT:    </main>
