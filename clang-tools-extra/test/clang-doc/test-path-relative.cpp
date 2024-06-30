// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --doxygen --executor=standalone -p %t %t/test.cpp -output=../docs
// RUN: FileCheck %s -input-file=%t/docs/index_json.js
// RUN: rm -rf %t

// CHECK: var RootPath = "{{.*}}../docs";