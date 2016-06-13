// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=133 -new-name=X %t.cpp -i --
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla {
public:
  int getValue() const {
    return 0;
  }
};

int main() {
  void *C = new Cla();
  reinterpret_cast<const Cla *>(C)->getValue(); // CHECK: reinterpret_cast<const X *>
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of foo when changing
// this file.
