// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp "%s" "%t/test.cpp"
// RUN: echo "CHECK: var RootPath = \"%t/../docs\";" > %t/check.txt
// RUN: clang-doc --format=html --executor=standalone -p %t %t/test.cpp --output=../docs
// RUN: FileCheck %t/check.txt -input-file=%t/../docs/index_json.js
// RUN: rm -rf %t