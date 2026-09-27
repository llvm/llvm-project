// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/../Inputs/nested-pointer-qualifiers.cpp
// RUN: FileCheck %s --check-prefix=HTML < %t/html/GlobalNamespace/index.html

// HTML: <pre><code class="language-cpp code-clang-doc">void foo (const int *const * ptr)</code></pre>
