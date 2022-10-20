// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

int strlen(char const *);

struct [[gsl::Owner(char *)]] String {
  long size;
  long capacity;
  const char *storage;

  String() : size{0}, capacity{0} {}
  String(char const *s) : size{strlen(s)}, capacity{size}, storage{s} {}
};

struct [[gsl::Pointer(int)]] StringView {
  long size;
  const char *storage;

  StringView(const String &s) : size{s.size}, storage{s.storage} {}
  StringView() : size{0}, storage{nullptr} {}
  int getSize() const;
};

void lifetime_example() {
  StringView sv;
  String name = "abcdefghijklmnop";
  sv = name;
  (void)sv.getSize(); // expected-remark {{pset => { name__1' }}}
  name = "frobozz"; // expected-note {{invalidated by non-const use of owner type}}
  (void)sv.getSize(); // expected-warning {{use of invalid pointer 'sv'}}
  // expected-remark@-1 {{pset => { invalid }}}
  sv = name;
  (void)sv.getSize(); // expected-remark {{pset => { name__2' }}}
}