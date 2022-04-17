// RUN: %clang_analyze_cc1 -std=c++14 -analyzer-checker=cplusplus.InnerPointer \
// RUN:   -Wno-dangling -Wno-dangling-field -Wno-return-stack-address \
// RUN:   %s -analyzer-output=text -verify

#include "Inputs/system-header-simulator-cxx.h"
namespace std {

template <typename T>
void func_ref(T &a);

template <typename T>
void func_const_ref(const T &a);

template <typename T>
void func_value(T a);

string my_string = "default";
void default_arg(int a = 42, string &b = my_string);

template <class T>
T *addressof(T &arg);

char *data(std::string &c);

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
    c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
    d = s.data();  // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  }                // expected-note {{Inner buffer of 'std::string' deallocated by call to destructor}}
  // expected-note@-1 {{Inner buffer of 'std::string' deallocated by call to destructor}}
  std::string s;
  const char *c2 = s.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is true}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is false}}
    // expected-note@-4 {{Taking false branch}}
    consume(c); // expected-warning {{Inner pointer of container used after re/deallocation}}
    // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
  } else {
    consume(d); // expected-warning {{Inner pointer of container used after re/deallocation}}
    // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
  }
}

void deref_after_scope_char_data_non_const() {
  char *c;
  {
    std::string s;
    c = s.data(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  }               // expected-note {{Inner buffer of 'std::string' deallocated by call to destructor}}
  std::string s;
  char *c2 = s.data();
  consume(c); // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_scope_wchar_t(bool cond) {
  const wchar_t *c, *d;
  {
    std::wstring s;
    c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::wstring' obtained here}}
    d = s.data();  // expected-note {{Pointer to inner buffer of 'std::wstring' obtained here}}
  }                // expected-note {{Inner buffer of 'std::wstring' deallocated by call to destructor}}
  // expected-note@-1 {{Inner buffer of 'std::wstring' deallocated by call to destructor}}
  std::wstring s;
  const wchar_t *c2 = s.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is true}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is false}}
    // expected-note@-4 {{Taking false branch}}
    consume(c); // expected-warning {{Inner pointer of container used after re/deallocation}}
    // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
  } else {
    consume(d); // expected-warning {{Inner pointer of container used after re/deallocation}}
    // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
  }
}

void deref_after_scope_char16_t_cstr() {
  const char16_t *c16;
  {
    std::u16string s16;
    c16 = s16.c_str(); // expected-note {{Pointer to inner buffer of 'std::u16string' obtained here}}
  }                    // expected-note {{Inner buffer of 'std::u16string' deallocated by call to destructor}}
  std::u16string s16;
  const char16_t *c16_2 = s16.c_str();
  consume(c16); // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_scope_char32_t_data() {
  const char32_t *c32;
  {
    std::u32string s32;
    c32 = s32.data(); // expected-note {{Pointer to inner buffer of 'std::u32string' obtained here}}
  }                   // expected-note {{Inner buffer of 'std::u32string' deallocated by call to destructor}}
  std::u32string s32;
  const char32_t *c32_2 = s32.data();
  consume(c32); // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void multiple_symbols(bool cond) {
  const char *c1, *d1;
  {
    std::string s1;
    c1 = s1.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
    d1 = s1.data();  // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
    const char *local = s1.c_str();
    consume(local); // no-warning
  }                 // expected-note {{Inner buffer of 'std::string' deallocated by call to destructor}}
  // expected-note@-1 {{Inner buffer of 'std::string' deallocated by call to destructor}}
  std::string s2;
  const char *c2 = s2.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is true}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is false}}
    // expected-note@-4 {{Taking false branch}}
    consume(c1); // expected-warning {{Inner pointer of container used after re/deallocation}}
    // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
  } else {
    consume(d1); // expected-warning {{Inner pointer of container used after re/deallocation}}
  }              // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
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
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s = "world";   // expected-note {{Inner buffer of 'std::string' reallocated by call to 'operator='}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_plus_equals() {
  const char *c;
  std::string s = "hello";
  c = s.data();  // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s += " world"; // expected-note {{Inner buffer of 'std::string' reallocated by call to 'operator+='}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_clear() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.clear();     // expected-note {{Inner buffer of 'std::string' reallocated by call to 'clear'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_append() {
  const char *c;
  std::string s = "hello";
  c = s.c_str();    // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.append(2, 'x'); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'append'}}
  consume(c);       // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_assign() {
  const char *c;
  std::string s;
  c = s.data();     // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.assign(4, 'a'); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'assign'}}
  consume(c);       // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_erase() {
  const char *c;
  std::string s = "hello";
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.erase(0, 2); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'erase'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_insert() {
  const char *c;
  std::string s = "ello";
  c = s.c_str();       // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.insert(0, 1, 'h'); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'insert'}}
  consume(c);          // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_replace() {
  const char *c;
  std::string s = "hello world";
  c = s.c_str();             // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.replace(6, 5, "string"); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'replace'}}
  consume(c);                // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_pop_back() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.pop_back();  // expected-note {{Inner buffer of 'std::string' reallocated by call to 'pop_back'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_push_back() {
  const char *c;
  std::string s;
  c = s.data();     // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.push_back('c'); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'push_back'}}
  consume(c);       // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_reserve() {
  const char *c;
  std::string s;
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.reserve(5);  // expected-note {{Inner buffer of 'std::string' reallocated by call to 'reserve'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_resize() {
  const char *c;
  std::string s;
  c = s.data(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.resize(5);  // expected-note {{Inner buffer of 'std::string' reallocated by call to 'resize'}}
  consume(c);   // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_shrink_to_fit() {
  const char *c;
  std::string s;
  c = s.data();      // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.shrink_to_fit(); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'shrink_to_fit'}}
  consume(c);        // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_swap() {
  const char *c;
  std::string s1, s2;
  c = s1.data(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s1.swap(s2);   // expected-note {{Inner buffer of 'std::string' reallocated by call to 'swap'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void deref_after_std_data() {
  const char *c;
  std::string s;
  c = std::data(s); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  s.push_back('c'); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'push_back'}}
  consume(c);       // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

struct S {
  std::string s;
  const char *name() {
    return s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
                      // expected-note@-1 {{Pointer to inner buffer of 'std::string' obtained here}}
  }
  void clear() {
    s.clear(); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'clear'}}
  }
  ~S() {} // expected-note {{Inner buffer of 'std::string' deallocated by call to destructor}}
};

void cleared_through_method() {
  S x;
  const char *c = x.name(); // expected-note {{Calling 'S::name'}}
                            // expected-note@-1 {{Returning from 'S::name'}}
  x.clear(); // expected-note {{Calling 'S::clear'}}
             // expected-note@-1 {{Returning; inner buffer was reallocated}}
  consume(c); // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void destroyed_through_method() {
  S y;
  const char *c = y.name(); // expected-note {{Calling 'S::name'}}
                            // expected-note@-1 {{Returning from 'S::name'}}
  y.~S(); // expected-note {{Calling '~S'}}
          // expected-note@-1 {{Returning; inner buffer was deallocated}}
  consume(c); // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

//=---------------------------=//
//     Other STL functions     //
//=---------------------------=//

void STL_func_ref() {
  const char *c;
  std::string s;
  c = s.c_str();    // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  std::func_ref(s); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'func_ref'}}
  consume(c);       // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
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
  c = s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  func_ptr(s);   // expected-note {{Inner buffer of 'std::string' reallocated by call to 'func_ref'}}
  consume(c);    // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
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
  c = s.c_str();     // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  default_arg(3, s); // expected-note {{Inner buffer of 'std::string' reallocated by call to 'default_arg'}}
  consume(c);        // expected-warning {{Inner pointer of container used after re/deallocation}}
  // expected-note@-1 {{Inner pointer of container used after re/deallocation}}
}

void func_addressof() {
  const char *c;
  std::string s;
  c = s.c_str();
  (void)addressof(s);
  consume(c); // no-warning
}

void func_std_data() {
  const char *c;
  std::string s;
  c = std::data(s);
  consume(c); // no-warning
}

struct T {
  std::string to_string() { return s; }

private:
  std::string s;
};

const char *escape_via_return_temp() {
  T x;
  return x.to_string().c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
  // expected-note@-1 {{Inner buffer of 'std::string' deallocated by call to destructor}}
  // expected-warning@-2 {{Inner pointer of container used after re/deallocation}}
  // expected-note@-3 {{Inner pointer of container used after re/deallocation}}
}

const char *escape_via_return_local() {
  std::string s;
  return s.c_str(); // expected-note {{Pointer to inner buffer of 'std::string' obtained here}}
                    // expected-note@-1 {{Inner buffer of 'std::string' deallocated by call to destructor}}
                    // expected-warning@-2 {{Inner pointer of container used after re/deallocation}}
                    // expected-note@-3 {{Inner pointer of container used after re/deallocation}}
}


char *c();
class A {};

void no_CXXRecordDecl() {
  A a, *b;
  *(void **)&b = c() + 1;
  *b = a; // no-crash
}

void checkReference(std::string &s) {
  const char *c = s.c_str();
}
