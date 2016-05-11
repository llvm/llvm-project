// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=162 -new-name=hector %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class A
{
};

class Cla
{
  A foo; // CHECK: hector;
public:
  Cla();
};

Cla::Cla() // CHECK: Cla::Cla()
{
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
