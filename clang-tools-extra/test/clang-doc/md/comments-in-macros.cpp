// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=md --doxygen --output=%t --executor=standalone %S/../Inputs/comments-in-macros.cpp
// RUN: clang-doc --format=md_mustache --doxygen --output=%t --executor=standalone %S/../Inputs/comments-in-macros.cpp
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS-LINE
// RUN: FileCheck %s < %t/GlobalNamespace/MyClass.md --check-prefix=MD-MYCLASS
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7MyClass.md --check-prefix=MD-MUSTACHE-MYCLASS-LINE
// RUN: FileCheck %s < %t/md/GlobalNamespace/_ZTV7MyClass.md --check-prefix=MD-MUSTACHE-MYCLASS

// MD-MYCLASS: ### Add
// MD-MYCLASS: *public int Add(int a, int b)*
// MD-MYCLASS: **brief** Declare a method to calculate the sum of two numbers

// MD-MUSTACHE-MYCLASS: ### Add
// MD-MUSTACHE-MYCLASS: *public int Add(int a, int b)*
// MD-MUSTACHE-MYCLASS: **brief** Declare a method to calculate the sum of two numbers

// MD-MYCLASS-LINE: *Defined at {{.*}}comments-in-macros.cpp#7*
// MD-MUSTACHE-MYCLASS-LINE: *Defined at {{.*}}comments-in-macros.cpp#7*
