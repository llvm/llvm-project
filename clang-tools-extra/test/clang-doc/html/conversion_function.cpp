// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --output=%t --executor=standalone %S/../Inputs/conversion_function.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV8MyStruct.html --check-prefix=CHECK-HTML

// Output correct conversion names.

// CHECK-HTML: <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML:     <pre><code class="language-cpp code-clang-doc">T operator T ()</code></pre>
// CHECK-HTML: </div>
