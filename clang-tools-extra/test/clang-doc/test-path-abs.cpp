// RUN: rm -rf %t && mkdir %t
// RUN: echo "CHECK: var RootPath = \"%/t/docs\";" > %t/check.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --format=html --executor=standalone -p %s --output=%t
// RUN: FileCheck %s -input-file=%t/index_json.js
// RUN: rm -rf %t