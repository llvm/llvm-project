// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=mustache --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/MyNamespace/index.html

namespace MyNamespace {
  class Foo;
}

// CHECK:       <ul class="class-container">
// CHECK-NEXT:    <li id="{{[0-9A-F]*}}" style="max-height: 40px;">
// CHECK-NEXT:        <a href="Foo.html"><pre><code class="language-cpp code-clang-doc" >class Foo</code></pre></a>
// CHECK-NEXT:    </li>
// CHECK-NEXT: </ul>
