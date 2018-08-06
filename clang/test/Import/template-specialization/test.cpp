// RUN: clang-import-test -import %S/Inputs/T.cpp -expression %s

void expr() {
  A<int>::B b1;
  A<bool>::B b2;
  b1.f + b2.g;
}

static_assert(f<char>() == 0, "");
static_assert(f<int>() == 4, "");
