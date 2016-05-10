// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=150 -new-name=hector %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla
{
  int foo; // CHECK: hector;
public:
  Cla();
};

Cla::Cla()
  : foo(0) // CHECK: hector(0)
{
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
