// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

struct mystruct {
  int  member;
};

template <int i>
int foo() {
  mystruct s[1];
  return s->member;
}

int main() {
  foo<1>();
}

// PR7405
struct hb_sanitize_context_t {
  int start;
};
template <typename Type> static bool sanitize() {
  hb_sanitize_context_t c[1];
  return !c->start;
}
bool closure = sanitize<int>();
