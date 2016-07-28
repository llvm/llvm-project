// RUN: clang-rename -offset=73 -new-name=X %s -- | FileCheck %s

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

// Use grep -FUbo 'Cla' <file> to get the correct offset of Cla when changing
// this file.
