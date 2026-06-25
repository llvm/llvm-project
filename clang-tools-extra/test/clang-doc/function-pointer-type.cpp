// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/Inputs/function-pointer-type.cpp
// RUN: FileCheck %s --check-prefix=HTML < %t/html/GlobalNamespace/index.html
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/Inputs/function-pointer-type.cpp
// RUN: FileCheck %s --check-prefix=MD < %t/GlobalNamespace/index.md
// RUN: clang-doc --output=%t --format=md_mustache --executor=standalone %S/Inputs/function-pointer-type.cpp
// RUN: FileCheck %s --check-prefix=MD-MUSTACHE < %t/md/GlobalNamespace/index.md

// HTML: <pre><code class="language-cpp code-clang-doc">void bar (void (*)(int) fn)</code></pre>

// MD: *void bar(void (*)(int) fn)*
// MD-MUSTACHE: *void bar(void (*)(int) fn)*
