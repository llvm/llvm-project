// RUN: cat %s > %t.cpp
// RUN: clang-rename -old-name=Cla -new-name=Hector %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla { // CHECK: class Hector
};

int main() {
  Cla *Pointer = 0; // CHECK: Hector *Pointer = 0;
  return 0;
}
