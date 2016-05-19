// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=133 -new-name=D %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class C
{
public:
    C();
};

C::C() // CHECK: D::D()
{
}

// Use grep -FUbo 'C' <file> to get the correct offset of foo when changing
// this file.
