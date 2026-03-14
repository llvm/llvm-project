// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

// Ensure that we don't crash if errors are suppressed by an error limit.
// RUN: not %clang_cc1 -fsyntax-only -std=c++17 -ferror-limit 1 %s

template <bool is_const, typename tag_t = void>
struct tv_val {
};

template <bool is_const>
auto &val(const tv_val<is_const> &val) { return val.val(); } // expected-note {{possible target for call}}

struct Class {
  template <bool is_const>
  struct Entry {
    tv_val<is_const> val;
  };
};

enum Types : int {
  Class = 1, // expected-note 2 {{struct 'Class' is hidden}}
};

struct Record {
  Class *val_;            // expected-error {{must use 'struct' tag}}
  void setClass(Class *); // expected-error {{must use 'struct' tag}}
};

void Record::setClass(Class *val) { // expected-error {{variable has incomplete type 'void'}} \
                                   // expected-error {{reference to overloaded function}} \
                                   // expected-error {{expected ';' after top level declarator}}
  val_ = val;
}
