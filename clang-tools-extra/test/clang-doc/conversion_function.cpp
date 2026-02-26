// RUN: rm -rf %t && mkdir -p %t

// RUN: clang-doc --output=%t --executor=standalone %s 
// RUN: find %t/ -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK-YAML

// RUN: clang-doc --format=html --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV8MyStruct.html --check-prefix=CHECK-HTML

template <typename T>
struct MyStruct {
  operator T();
};

// Output correct conversion names.
// CHECK-YAML:         Name:            'operator T'

// CHECK-HTML: <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML:     <pre><code class="language-cpp code-clang-doc">T operator T ()</code></pre>
// CHECK-HTML: </div>
