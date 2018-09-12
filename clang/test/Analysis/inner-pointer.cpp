//RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus.InnerPointer %s -analyzer-output=text -verify

namespace std {

typedef int size_type;

template <typename CharT>
class basic_string {
public:
  basic_string();
  basic_string(const CharT *s);

  ~basic_string();
  void clear();

  basic_string &operator=(const basic_string &str);
  basic_string &operator+=(const basic_string &str);

  const CharT *c_str() const;
  const CharT *data() const;
  CharT *data();

  basic_string &append(size_type count, CharT ch);
  basic_string &assign(size_type count, CharT ch);
  basic_string &erase(size_type index, size_type count);
  basic_string &insert(size_type index, size_type count, CharT ch);
  basic_string &replace(size_type pos, size_type count, const basic_string &str);
  void pop_back();
  void push_back(CharT ch);
  void reserve(size_type new_cap);
  void resize(size_type count);
  void shrink_to_fit();
  void swap(basic_string &other);
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
typedef basic_string<char16_t> u16string;
typedef basic_string<char32_t> u32string;

template <typename T>
void func_ref(T &a);

template <typename T>
void func_const_ref(const T &a);

template <typename T>
void func_value(T a);

string my_string = "default";
void default_arg(int a = 42, string &b = my_string);

} // end namespace std

void consume(const char *) {}
void consume(const wchar_t *) {}
void consume(const char16_t *) {}
void consume(const char32_t *) {}

//=--------------------------------------=//
//     `std::string` member functions     //
//=--------------------------------------=//

void deref_after_scope_char(bool cond) {
  const char *c, *d;
  {
    std::string s;
    c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
    d = s.data();  // expected-note {{Dangling inner pointer obtained here}}
  }                // expected-note {{Inner pointer invalidated by call to destructor}}
  // expected-note@-1 {{Inner pointer invalidated by call to destructor}}
  std::string s;
  const char *c2 = s.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is 0}}
    // expected-note@-4 {{Taking false branch}}
    consume(c); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  } else {
    consume(d); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  }
}

void deref_after_scope_char_data_non_const() {
  char *c;
  {
    std::string s;
    c = s.data(); // expected-note {{Dangling inner pointer obtained here}}
  }               // expected-note {{Inner pointer invalidated by call to destructor}}
  std::string s;
  char *c2 = s.data();
  consume(c); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_wchar_t(bool cond) {
  const wchar_t *c, *d;
  {
    std::wstring s;
    c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
    d = s.data();  // expected-note {{Dangling inner pointer obtained here}}
  }                // expected-note {{Inner pointer invalidated by call to destructor}}
  // expected-note@-1 {{Inner pointer invalidated by call to destructor}}
  std::wstring s;
  const wchar_t *c2 = s.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is 0}}
    // expected-note@-4 {{Taking false branch}}
    consume(c); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  } else {
    consume(d); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  }
}

void deref_after_scope_char16_t_cstr() {
  const char16_t *c16;
  {
    std::u16string s16;
    c16 = s16.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  }                    // expected-note {{Inner pointer invalidated by call to destructor}}
  std::u16string s16;
  const char16_t *c16_2 = s16.c_str();
  consume(c16); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char32_t_data() {
  const char32_t *c32;
  {
    std::u32string s32;
    c32 = s32.data(); // expected-note {{Dangling inner pointer obtained here}}
  }                   // expected-note {{Inner pointer invalidated by call to destructor}}
  std::u32string s32;
  const char32_t *c32_2 = s32.data();
  consume(c32); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void multiple_symbols(bool cond) {
  const char *c1, *d1;
  {
    std::string s1;
    c1 = s1.c_str(); // expected-note {{Dangling inner pointer obtained here}}
    d1 = s1.data();  // expected-note {{Dangling inner pointer obtained here}}
    const char *local = s1.c_str();
    consume(local); // no-warning
  }                 // expected-note {{Inner pointer invalidated by call to destructor}}
  // expected-note@-1 {{Inner pointer invalidated by call to destructor}}
  std::string s2;
  const char *c2 = s2.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is 0}}
    // expected-note@-4 {{Taking false branch}}
    consume(c1); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  } else {
    consume(d1); // expected-warning {{Use of memory after it is freed}}
  }              // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_ok(bool cond) {
  const char *c, *d;
  std::string s;
  {
    c = s.c_str();
    d = s.data();
  }
  if (cond)
    consume(c); // no-warning
  else
    consume(d); // no-warning
}

void deref_after_equals() {
  const char *c;
  std::string s = "hello";
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  s = "world";   // expected-note {{Inner pointer invalidated by call to 'operator='}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_plus_equals() {
  const char *c;
  std::string s = "hello";
  c = s.data();  // expected-note {{Dangling inner pointer obtained here}}
  s += " world"; // expected-note {{Inner pointer invalidated by call to 'operator+='}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_clear() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  s.clear();     // expected-note {{Inner pointer invalidated by call to 'clear'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_append() {
  const char *c;
  std::string s = "hello";
  c = s.c_str();    // expected-note {{Dangling inner pointer obtained here}}
  s.append(2, 'x'); // expected-note {{Inner pointer invalidated by call to 'append'}}
  consume(c);       // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_assign() {
  const char *c;
  std::string s;
  c = s.data();     // expected-note {{Dangling inner pointer obtained here}}
  s.assign(4, 'a'); // expected-note {{Inner pointer invalidated by call to 'assign'}}
  consume(c);       // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_erase() {
  const char *c;
  std::string s = "hello";
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  s.erase(0, 2); // expected-note {{Inner pointer invalidated by call to 'erase'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_insert() {
  const char *c;
  std::string s = "ello";
  c = s.c_str();       // expected-note {{Dangling inner pointer obtained here}}
  s.insert(0, 1, 'h'); // expected-note {{Inner pointer invalidated by call to 'insert'}}
  consume(c);          // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_replace() {
  const char *c;
  std::string s = "hello world";
  c = s.c_str();             // expected-note {{Dangling inner pointer obtained here}}
  s.replace(6, 5, "string"); // expected-note {{Inner pointer invalidated by call to 'replace'}}
  consume(c);                // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_pop_back() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  s.pop_back();  // expected-note {{Inner pointer invalidated by call to 'pop_back'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_push_back() {
  const char *c;
  std::string s;
  c = s.data();     // expected-note {{Dangling inner pointer obtained here}}
  s.push_back('c'); // expected-note {{Inner pointer invalidated by call to 'push_back'}}
  consume(c);       // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_reserve() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  s.reserve(5);  // expected-note {{Inner pointer invalidated by call to 'reserve'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_resize() {
  const char *c;
  std::string s;
  c = s.data(); // expected-note {{Dangling inner pointer obtained here}}
  s.resize(5);  // expected-note {{Inner pointer invalidated by call to 'resize'}}
  consume(c);   // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_shrink_to_fit() {
  const char *c;
  std::string s;
  c = s.data();      // expected-note {{Dangling inner pointer obtained here}}
  s.shrink_to_fit(); // expected-note {{Inner pointer invalidated by call to 'shrink_to_fit'}}
  consume(c);        // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_swap() {
  const char *c;
  std::string s1, s2;
  c = s1.data(); // expected-note {{Dangling inner pointer obtained here}}
  s1.swap(s2);   // expected-note {{Inner pointer invalidated by call to 'swap'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

//=---------------------------=//
//     Other STL functions     //
//=---------------------------=//

void STL_func_ref() {
  const char *c;
  std::string s;
  c = s.c_str();    // expected-note {{Dangling inner pointer obtained here}}
  std::func_ref(s); // expected-note {{Inner pointer invalidated by call to 'func_ref'}}
  consume(c);       // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void STL_func_const_ref() {
  const char *c;
  std::string s;
  c = s.c_str();
  std::func_const_ref(s);
  consume(c); // no-warning
}

void STL_func_value() {
  const char *c;
  std::string s;
  c = s.c_str();
  std::func_value(s);
  consume(c); // no-warning
}

void func_ptr_known() {
  const char *c;
  std::string s;
  void (*func_ptr)(std::string &) = std::func_ref<std::string>;
  c = s.c_str(); // expected-note {{Dangling inner pointer obtained here}}
  func_ptr(s);   // expected-note {{Inner pointer invalidated by call to 'func_ref'}}
  consume(c);    // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void func_ptr_unknown(void (*func_ptr)(std::string &)) {
  const char *c;
  std::string s;
  c = s.c_str();
  func_ptr(s);
  consume(c); // no-warning
}

void func_default_arg() {
  const char *c;
  std::string s;
  c = s.c_str();     // expected-note {{Dangling inner pointer obtained here}}
  default_arg(3, s); // expected-note {{Inner pointer invalidated by call to 'default_arg'}}
  consume(c);        // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}
