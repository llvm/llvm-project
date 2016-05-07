class Cla
{
  int foo; // CHECK: hector;
public:
  Cla();
};
// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=18 -new-name=hector %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

Cla::Cla()
  : foo(0) // CHECK: hector(0)
{
}

// Use grep -FUbo 'foo' <file> to get the correct offset of foo when changing
// this file.
