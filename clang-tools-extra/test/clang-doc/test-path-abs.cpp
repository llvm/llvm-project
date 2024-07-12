// RUN: rm -rf %t && mkdir %t
// RUN: echo "CHECK: var RootPath = \"%/t/docs\";" > %t/check.txt
// RUN: cp "%s" "%t/test.cpp"
// RUN: clang-doc --format=html --executor=standalone -p %t %t/test.cpp --output=%t/docs
// RUN: FileCheck %t/check.txt -input-file=%t/docs/index_json.js
// RUN: rm -rf %t