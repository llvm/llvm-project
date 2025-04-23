// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --executor=standalone %s --output=%t --base base_dir
// RUN: FileCheck %s -input-file=%t/index_json.js  -check-prefix=JSON-INDEX

// JSON-INDEX: var RootPath = "{{.*}}test-path-abs.cpp.tmp";
// JSON-INDEX-NEXT: var Base = "base_dir";

