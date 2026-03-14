// RUN: clang-import-test -import %S/Inputs/I1.cpp -expression %s

void expr() {
  f(0);
}

int std = 17;
