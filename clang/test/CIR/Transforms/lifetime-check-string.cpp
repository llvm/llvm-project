// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

int strlen(char const *);

struct [[gsl::Owner(char *)]] String {
  long size;
  long capacity;
  const char *storage;
  char operator[](int);
  String() : size{0}, capacity{0} {}
  String(char const *s) : size{strlen(s)}, capacity{size}, storage{s} {}
};

struct [[gsl::Pointer(int)]] StringView {
  long size;
  const char *storage;
  char operator[](int);
  StringView(const String &s) : size{s.size}, storage{s.storage} {}
  StringView() : size{0}, storage{nullptr} {}
  int getSize() const;
};

void sv0() {
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

void sv1() {
  StringView sv, sv_other;
  String name = "abcdefghijklmnop";
  sv = name;
  sv_other = sv;
  (void)sv.getSize();  // expected-remark {{pset => { name__1' }}}
  (void)sv_other.getSize();  // expected-remark {{pset => { name__1' }}}
  name = "frobozz"; // expected-note {{invalidated by non-const use of owner type}}
  (void)sv.getSize(); // expected-warning {{use of invalid pointer 'sv'}}
  // expected-remark@-1 {{pset => { invalid }}}
  (void)sv_other.getSize(); // expected-warning {{use of invalid pointer 'sv_other'}}
  // expected-remark@-1 {{pset => { invalid }}}
  sv = name;
  (void)sv.getSize(); // expected-remark {{pset => { name__2' }}}
}

void sv2() {
  StringView sv;
  String name = "abcdefghijklmnop";
  sv = name;
  char read0 = sv[0]; // expected-remark {{pset => { name__1' }}}
  name = "frobozz"; // expected-note {{invalidated by non-const use of owner type}}
  char read1 = sv[0]; // expected-warning {{use of invalid pointer 'sv'}}
  // expected-remark@-1 {{pset => { invalid }}}
  sv = name;
  char read2 = sv[0]; // expected-remark {{pset => { name__2' }}}
  char read3 = name[1]; // expected-note {{invalidated by non-const use of owner type}}
  char read4 = sv[1]; // expected-warning {{use of invalid pointer 'sv'}}
  // expected-remark@-1 {{pset => { invalid }}}
}

class Stream {
 public:
  Stream& operator<<(char);
  Stream& operator<<(const StringView &);
  // FIXME: conservative for now, but do not invalidate const Owners?
  Stream& operator<<(const String &);
};

void sv3() {
  Stream cout;
  StringView sv;
  String name = "abcdefghijklmnop";
  sv = name;
  cout << sv; // expected-remark {{pset => { name__1' }}}
  name = "frobozz"; // expected-note {{invalidated by non-const use of owner type}}
  cout << sv[2]; // expected-warning {{use of invalid pointer 'sv'}}
  sv = name; // expected-remark@-1 {{pset => { invalid }}}
  cout << sv; // expected-remark {{pset => { name__2' }}}
  cout << name; // expected-note {{invalidated by non-const use of owner type}}
  cout << sv; // expected-warning {{passing invalid pointer 'sv'}}
  // expected-remark@-1 {{pset => { invalid }}}
}