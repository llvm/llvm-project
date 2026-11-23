// RUN: rm -rf %t && mkdir %t
// RUN: clang-doc --format=html --output=%t --asset=%S/Inputs/test-assets --executor=standalone %s --base base_dir
// RUN: FileCheck %s -input-file=%t/html/test.css -check-prefix=CSS
// RUN: FileCheck %s -input-file=%t/html/test.js -check-prefix=JS

// CSS: body {
// CSS-NEXT:     padding: 0;
// CSS-NEXT: }

// JS: console.log("Hello, world!");
