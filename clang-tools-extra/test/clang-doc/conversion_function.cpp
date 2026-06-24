// RUN: rm -rf %t && mkdir -p %t

// RUN: clang-doc --output=%t --executor=standalone %s 
// RUN: find %t/ -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK-YAML

// RUN: clang-doc --format=html --output=%t --executor=standalone %s 
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV8MyStruct.html --check-prefix=CHECK-HTML

// RUN: clang-doc --doxygen --format=md_mustache --output=%t --executor=standalone %s
// RUN: FileCheck %s --input-file=%t/md/GlobalNamespace/_ZTV8MyStruct.md --check-prefix=MD-MUSTACHE

template <typename T>
struct MyStruct {
  operator T();
};

// Output correct conversion names.
// CHECK-YAML:         Name:            'operator T'

// CHECK-HTML: <div id="{{([0-9A-F]{40})}}" class="delimiter-container">
// CHECK-HTML:     <pre><code class="language-cpp code-clang-doc">T operator T ()</code></pre>
// CHECK-HTML: </div>

// MD-MUSTACHE: # struct MyStruct
// MD-MUSTACHE: ## Functions
// MD-MUSTACHE: ### operator T
// MD-MUSTACHE: *public T operator T()*