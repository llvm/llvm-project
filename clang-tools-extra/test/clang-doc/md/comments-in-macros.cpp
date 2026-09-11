// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md -doxygen --output=%t --executor=standalone %S/../Inputs/comments-in-macros.cpp
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7MyClass.md --check-prefix=MD-MYCLASS-LINE
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7MyClass.md --check-prefix=MD-MYCLASS


// MD-MYCLASS: ### Add
// MD-MYCLASS: *public int Add(int a, int b)*
// MD-MYCLASS: **brief** Declare a method to calculate the sum of two numbers

// MD-MYCLASS-LINE: *Defined at {{.*}}comments-in-macros.cpp#7*
