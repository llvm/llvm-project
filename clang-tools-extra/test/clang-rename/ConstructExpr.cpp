// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=133 -new-name=D %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla
{
};

int main()
{
  Cla *C = new Cla(); // CHECK: D *C = new D();
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of foo when changing
// this file.
