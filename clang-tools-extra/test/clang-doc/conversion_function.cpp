// RUN: rm -rf %t && mkdir -p %t

// RUN: clang-doc --output=%t --executor=standalone %s 
// RUN: find %t/ -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK-YAML

// RUN: clang-doc --format=html --output=%t --executor=standalone %s 
// FileCheck %s --check-prefix=CHECK-HTML

template <typename T>
struct MyStruct {
  operator T();
};

// Output incorrect conversion names.
// CHECK-YAML:         Name:            'operator type-parameter-0-0'
// CHECK-YAML-NOT:     Name:            'operator T'

// CHECK-HTML-NOT: <h3 id='{{[0-9A-F]*}}'>operator T</h3>
// CHECK-HTML-NOT: <p>public T operator T()</p>
