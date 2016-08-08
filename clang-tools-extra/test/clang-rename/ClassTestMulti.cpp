// RUN: cat %s > %t.cpp
// RUN: clang-rename rename-all -offset=174 -new-name=Bar1 -offset=212 -new-name=Bar2 %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Foo1 { // CHECK: class Bar1
};

class Foo2 { // CHECK: class Bar2
};
