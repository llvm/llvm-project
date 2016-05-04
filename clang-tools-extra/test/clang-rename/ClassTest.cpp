class Cla  // CHECK: class Hector
{
};
// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=6 -new-name=Hector %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

int main()
{
    Cla *Pointer = 0; // CHECK: Hector *Pointer = 0;
    return 0;
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of Cla when changing
// this file.
