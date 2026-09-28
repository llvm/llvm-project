// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -verify %s

int main() {
  int i;
  return ({
    struct T {
    } s[-sizeof(0)][0 == sizeof(i < 0)]; // expected-error {{array is too large (18'446'744'073'709'551'612 elements)}}
    0;
  });
}

int original() {
  int i = 0;
  return 1 + ({
    struct tree_el {
      int val;
      struct tree_el **right, *left;
    } state_t[1 + -(sizeof(0x1c))][0 == sizeof(sizeof(i))]; // expected-error {{array is too large (18'446'744'073'709'551'613 elements)}}
    0x97 < 10000;
  });
}
