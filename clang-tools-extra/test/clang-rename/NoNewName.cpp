// This test is a copy of ConstCastExpr.cpp with a single change:
// -new-name hasn't been passed to clang-rename, so this test should give an
// error.
// RUN: not clang-rename -offset=133 %s 2>&1 | FileCheck %s
// CHECK: clang-rename: no new name provided.

class Cla {
public:
  int getValue() {
    return 0;
  }
};

int main() {
  const Cla *C = new Cla();
  const_cast<Cla *>(C)->getValue();
}

// Use grep -FUbo 'Cla' <file> to get the correct offset of foo when changing
// this file.
