// RUN: cat %s > %t.cpp
// RUN: clang-rename rename-all -old-name=Foo1 -new-name=Bar1 -old-name=Foo2 -new-name=Bar2 %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Foo1 { // CHECK: class Bar1
};

class Foo2 { // CHECK: class Bar2
};
