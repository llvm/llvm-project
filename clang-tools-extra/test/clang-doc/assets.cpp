// RUN: rm -rf %t && mkdir %t
// RUN: clang-doc --format=html --output=%t --asset=%S/Inputs/test-assets --executor=standalone %s
// RUN: FileCheck %s -input-file=%t/index.html -check-prefix=INDEX
// RUN: FileCheck %s -input-file=%t/test.css -check-prefix=CSS
// RUN: FileCheck %s -input-file=%t/test.js -check-prefix=JS

// INDEX: <!DOCTYPE html>
// INDEX-NEXT: <meta charset="utf-8"/>
// INDEX-NEXT: <title>Index</title>
// INDEX-NEXT: <link rel="stylesheet" href="test.css"/>
// INDEX-NEXT: <script src="index_json.js"></script>
// INDEX-NEXT: <script src="test.js"></script>
// INDEX-NEXT: <header id="project-title"></header>
// INDEX-NEXT: <main>
// INDEX-NEXT:   <div id="sidebar-left" path="" class="col-xs-6 col-sm-3 col-md-2 sidebar sidebar-offcanvas-left" style="flex: 0 100%;"></div>
// INDEX-NEXT: </main>

// CSS: body {
// CSS-NEXT:     padding: 0;
// CSS-NEXT: }

// JS: console.log("Hello, world!");