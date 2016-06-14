// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=133 -new-name=X %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla {
public:
  int getValue() {
    return 0;
  }
};

int main() {
  const Cla *C = new Cla();
  const_cast<Cla *>(C)->getValue(); // CHECK: const_cast<X *>
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of foo when changing
// this file.
