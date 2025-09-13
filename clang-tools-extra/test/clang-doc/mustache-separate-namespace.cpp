// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=mustache --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/MyNamespace.html

namespace MyNamespace {
  class Foo;
}

// CHECK:       <ul class="class-container">
// CHECK-NEXT:    <li id="{{[0-9A-F]*}}" style="max-height: 40px;">
// CHECK-NEXT:        <a href="_ZTVN11MyNamespace3FooE.html">
// CHECK-NEXT:            <pre>
// CHECK-NEXT:                <code class="language-cpp code-clang-doc">class Foo</code>
// CHECK-NEXT:            </pre>
// CHECK-NEXT:        </a>
// CHECK-NEXT:    </li>
// CHECK-NEXT: </ul>
