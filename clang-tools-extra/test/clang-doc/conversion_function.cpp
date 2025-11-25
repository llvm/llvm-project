// RUN: rm -rf %t && mkdir -p %t

// RUN: clang-doc --output=%t --executor=standalone %s 
// RUN: find %t/ -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK-YAML

// RUN: clang-doc --format=html --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/GlobalNamespace/MyStruct.html --check-prefix=CHECK-HTML

// RUN: clang-doc --format=mustache --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV8MyStruct.html --check-prefix=CHECK-MUSTACHE

template <typename T>
struct MyStruct {
  operator T();
};

// Output correct conversion names.
// CHECK-YAML:         Name:            'operator T'

// CHECK-HTML: <h3 id="{{[0-9A-F]*}}">operator T</h3>
// CHECK-HTML: <p>public T operator T()</p>

// CHECK-MUSTACHE: <div id="{{([0-9A-F]{40})}}">
// CHECK-MUSTACHE:     <pre><code class="language-cpp code-clang-doc">T operator T ()</code></pre>
// CHECK-MUSTACHE: </div>
