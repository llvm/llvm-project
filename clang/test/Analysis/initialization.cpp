// RUN: %clang_analyze_cc1 -std=c++14 -triple i386-apple-darwin10 -analyzer-config eagerly-assume=false -analyzer-checker=core.uninitialized.Assign,core.builtin,debug.ExprInspection,core.uninitialized.UndefReturn -verify %s

template <typename T>
void clang_analyzer_dump(T x);
void clang_analyzer_eval(int);

extern const int glob_arr_no_init[10];
void glob_array_index4() {
  clang_analyzer_eval(glob_arr_no_init[2]); // expected-warning{{UNKNOWN}}
}

struct S2 {
  static const int arr_no_init[10];
};
void struct_arr_index1() {
  clang_analyzer_eval(S2::arr_no_init[2]); // expected-warning{{UNKNOWN}}
}

char const glob_arr6[5] = "123";
void glob_array_index5() {
  clang_analyzer_eval(glob_arr6[0] == '1');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr6[1] == '2');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr6[2] == '3');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr6[3] == '\0'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr6[4] == '\0'); // expected-warning{{TRUE}}
}

void glob_ptr_index3() {
  char const *ptr = glob_arr6;
  clang_analyzer_eval(ptr[-42] == '\0'); // expected-warning{{UNDEFINED}}
  clang_analyzer_eval(ptr[0] == '1');    // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[1] == '2');    // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[2] == '3');    // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[3] == '\0');   // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[4] == '\0');   // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[5] == '\0');   // expected-warning{{UNDEFINED}}
  clang_analyzer_eval(ptr[6] == '\0');   // expected-warning{{UNDEFINED}}
}

void glob_invalid_index7() {
  int idx = -42;
  auto x = glob_arr6[idx]; // expected-warning{{uninitialized}}
}

void glob_invalid_index8() {
  const char *ptr = glob_arr6;
  int idx = 42;
  auto x = ptr[idx]; // expected-warning{{uninitialized}}
}

char const *const glob_ptr8 = "123";
void glob_ptr_index4() {
  clang_analyzer_eval(glob_ptr8[0] == '1');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr8[1] == '2');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr8[2] == '3');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr8[3] == '\0'); // expected-warning{{TRUE}}
  // FIXME: Should be UNDEFINED.
  // We should take into account a declaration in which the literal is used.
  clang_analyzer_eval(glob_ptr8[4] == '\0'); // expected-warning{{TRUE}}
}

void glob_invalid_index11() {
  int idx = -42;
  auto x = glob_ptr8[idx]; // expected-warning{{uninitialized}}
}

void glob_invalid_index12() {
  int idx = 42;
  // FIXME: Should warn {{uninitialized}}
  // We should take into account a declaration in which the literal is used.
  auto x = glob_ptr8[idx]; // no-warning
}

const char16_t *const glob_ptr9 = u"абв";
void glob_ptr_index5() {
  clang_analyzer_eval(glob_ptr9[0] == u'а'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr9[1] == u'б'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr9[2] == u'в'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr9[3] == '\0'); // expected-warning{{TRUE}}
}

const char32_t *const glob_ptr10 = U"\U0001F607\U0001F608\U0001F609";
void glob_ptr_index6() {
  clang_analyzer_eval(glob_ptr10[0] == U'\U0001F607'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr10[1] == U'\U0001F608'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr10[2] == U'\U0001F609'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr10[3] == '\0');          // expected-warning{{TRUE}}
}

const wchar_t *const glob_ptr11 = L"\123\u0041\xFF";
void glob_ptr_index7() {
  clang_analyzer_eval(glob_ptr11[0] == L'\123');   // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr11[1] == L'\u0041'); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr11[2] == L'\xFF');   // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr11[3] == L'\0');     // expected-warning{{TRUE}}
}

const char *const glob_ptr12 = u8"abc";
void glob_ptr_index8() {
  clang_analyzer_eval(glob_ptr12[0] == 'a');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr12[1] == 'b');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr12[2] == 'c');  // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_ptr12[3] == '\0'); // expected-warning{{TRUE}}
}
