// RUN: rm -rf %t && mkdir %t
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --format=html --executor=standalone -p %s --output=%t
// RUN: FileCheck %s -input-file=%t/index_json.js  -check-prefix=JSON-INDEX
// RUN: rm -rf %t

// JSON-INDEX: var RootPath = "{{.*}}";