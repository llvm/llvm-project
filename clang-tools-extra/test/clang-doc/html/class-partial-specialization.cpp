// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/../Inputs/class-partial-specialization.cpp
// RUN: FileCheck %s --check-prefix=HTML < %t/html/GlobalNamespace/_ZTV7MyClassIPT_E.html

// HTML: <pre><code class="language-cpp code-clang-doc">template &lt;typename T&gt;</code></pre>
// HTML: <h1 class="hero__title-large">struct MyClass</h1>
