// Fixes #59819. The underlying problem was fixed in https://reviews.llvm.org/D142560, but this patch adds a proper regression test.
// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MyClass-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MyClass
// RUN: clang-doc --format=html --doxygen --output=%t --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.html --check-prefix=HTML-MyClass-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.html --check-prefix=HTML-MyClass

#define DECLARE_METHODS                                           \
    /**   							  \
     * @brief Declare a method to calculate the sum of two numbers\
     */                                                           \
    int Add(int a, int b)                                         \
    {                                                             \
        return a + b;                                             \
    }

// MD-MyClass: ### Add
// MD-MyClass: *public int Add(int a, int b)*
// MD-MyClass: **brief** Declare a method to calculate the sum of two numbers

// HTML-MyClass: <p>public int Add(int a, int b)</p>
// HTML-MyClass: <div>brief</div>
// HTML-MyClass: <p> Declare a method to calculate the sum of two numbers</p>


class MyClass {
public:
// MD-MyClass-LINE: *Defined at {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}macro.cpp#[[@LINE+2]]*
// HTML-MyClass-LINE: <p>Defined at line [[@LINE+1]] of file {{.*}}clang-tools-extra{{[\/]}}test{{[\/]}}clang-doc{{[\/]}}macro.cpp</p>
    DECLARE_METHODS
};


