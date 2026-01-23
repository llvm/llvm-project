// Regression test for https://github.com/llvm/llvm-project/issues/59819

// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS

// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html --check-prefix=HTML-MYCLASS-LINE
// RUN: FileCheck %s < %t/html/GlobalNamespace/_ZTV7MyClass.html --check-prefix=HTML-MYCLASS

#define DECLARE_METHODS                                           \
    /**   							  
     * @brief Declare a method to calculate the sum of two numbers
     */                                                           \
    int Add(int a, int b) {                                       \
        return a + b;                                             \
    }

// MD-MYCLASS: ### Add
// MD-MYCLASS: *public int Add(int a, int b)*
// MD-MYCLASS: **brief** Declare a method to calculate the sum of two numbers


// HTML-MYCLASS: <pre><code class="language-cpp code-clang-doc">int Add (int a, int b)</code></pre>
// HTML-MYCLASS: <div class="doc-card">
// HTML-MYCLASS:     <div class="nested-delimiter-container">
// HTML-MYCLASS:         <p> Declare a method to calculate the sum of two numbers</p>
// HTML-MYCLASS:     </div>


class MyClass {
public:
// MD-MYCLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}comments-in-macros.cpp#[[@LINE-2]]*
// HTML-MYCLASS-LINE: <p>Defined at line [[@LINE-3]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}comments-in-macros.cpp</p>
    DECLARE_METHODS
};

