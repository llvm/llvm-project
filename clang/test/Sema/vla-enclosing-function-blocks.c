// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

// GH55686
int size_tab(void);
int main(void) {
  __auto_type tab = (int(*)[size_tab()])0; // expected-note {{size expression is evaluated here}}
  (^ int(__typeof(tab) arr) { return sizeof(*arr); })(tab); // expected-error {{cannot be used in a block}}
}

void var_in_block(void) {
  typedef int X[size_tab()]; // expected-note {{size expression is evaluated here}}
  ^{ X x; }(); // expected-error {{cannot be used in a block}}
}

// Valid: VLA declared inside the block itself.
int own_vla_in_block(void) {
  return (^ int(void) { int a[size_tab()]; return sizeof(a); })();
}
