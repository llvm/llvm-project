// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %S/../Inputs/comments-in-macros.cpp
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html --check-prefix=HTML-MYCLASS-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html --check-prefix=HTML-MYCLASS

// HTML-MYCLASS: <pre><code class="language-cpp code-clang-doc">int Add (int a, int b)</code></pre>
// HTML-MYCLASS: <div class="doc-card">
// HTML-MYCLASS:     <div class="nested-delimiter-container">
// HTML-MYCLASS:         <p>Declare a method to calculate the sum of two numbers</p>
// HTML-MYCLASS:     </div>

// HTML-MYCLASS-LINE: <p>Defined at line 7 of file {{.*}}comments-in-macros.cpp</p>
