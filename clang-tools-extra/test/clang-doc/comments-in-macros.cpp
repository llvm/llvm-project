// Regression test for https://github.com/llvm/llvm-project/issues/59819

// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS

// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.html --check-prefix=HTML-MYCLASS-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.html --check-prefix=HTML-MYCLASS

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

// HTML-MYCLASS: <p>public int Add(int a, int b)</p>
// HTML-MYCLASS: <div>brief</div>
// HTML-MYCLASS: <p> Declare a method to calculate the sum of two numbers</p>


class MyClass {
public:
// MD-MYCLASS-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}comments-in-macros.cpp#[[@LINE+2]]*
// HTML-MYCLASS-LINE: <p>Defined at line [[@LINE+1]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}comments-in-macros.cpp</p>
    DECLARE_METHODS
};

