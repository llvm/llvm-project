// RUN: %clang_analyze_cc1 -std=c++11 %s -verify -analyzer-output=text \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=cplusplus \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

extern "C" char *strdup(const char* s);
extern "C" void free(void* ptr);

namespace std {
template<class T> struct remove_reference      { typedef T type; };
template<class T> struct remove_reference<T&>  { typedef T type; };
template<class T> struct remove_reference<T&&> { typedef T type; };
template<class T> typename remove_reference<T>::type&& move(T&& t);
}

void clang_analyzer_eval(int);

class StringUsed {
public:
  StringUsed(const char *s = "") : str(strdup(s)) {}
  StringUsed(const StringUsed &rhs) : str(strdup(rhs.str)) {}
  ~StringUsed();
  StringUsed& operator=(const StringUsed &rhs);
  StringUsed& operator=(StringUsed &&rhs);
  operator const char*() const;
private:
  char *str;
};

StringUsed::~StringUsed() {
  free(str);
}

StringUsed &StringUsed::operator=(const StringUsed &rhs) {
  // expected-note@-1{{Assuming rhs == *this}}
  // expected-note@-2{{Assuming rhs == *this}}
  // expected-note@-3{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{UNKNOWN}}
                                     // expected-note@-2{{TRUE}}
                                     // expected-note@-3{{UNKNOWN}}
  free(str); // expected-note{{Memory is released}}
  str = strdup(rhs.str); // expected-warning{{Use of memory after it is freed}}
                         // expected-note@-1{{Use of memory after it is freed}}
                         // expected-note@-2{{Memory is allocated}}
  return *this;
}

StringUsed &StringUsed::operator=(StringUsed &&rhs) {
  // expected-note@-1{{Assuming rhs == *this}}
  // expected-note@-2{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{UNKNOWN}}
                                     // expected-note@-2{{TRUE}}
                                     // expected-note@-3{{UNKNOWN}}
  str = rhs.str;
  rhs.str = nullptr; // expected-warning{{Potential memory leak}}
                     // expected-note@-1{{Potential memory leak}}
  return *this;
}

StringUsed::operator const char*() const {
  return str;
}

class StringUnused {
public:
  StringUnused(const char *s = "") : str(strdup(s)) {}
  StringUnused(const StringUnused &rhs) : str(strdup(rhs.str)) {}
  ~StringUnused();
  StringUnused& operator=(const StringUnused &rhs);
  StringUnused& operator=(StringUnused &&rhs);
  operator const char*() const;
private:
  char *str;
};

StringUnused::~StringUnused() {
  free(str);
}

StringUnused &StringUnused::operator=(const StringUnused &rhs) {
  // expected-note@-1{{Assuming rhs == *this}}
  // expected-note@-2{{Assuming rhs == *this}}
  // expected-note@-3{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{UNKNOWN}}
                                     // expected-note@-2{{TRUE}}
                                     // expected-note@-3{{UNKNOWN}}
  free(str); // expected-note{{Memory is released}}
  str = strdup(rhs.str); // expected-warning{{Use of memory after it is freed}}
                         // expected-note@-1{{Use of memory after it is freed}}
  return *this;
}

StringUnused &StringUnused::operator=(StringUnused &&rhs) {
  // expected-note@-1{{Assuming rhs == *this}}
  // expected-note@-2{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{UNKNOWN}}
                                     // expected-note@-2{{TRUE}}
                                     // expected-note@-3{{UNKNOWN}}
  str = rhs.str;
  rhs.str = nullptr; // FIXME: An improved leak checker should warn here
  return *this;
}

StringUnused::operator const char*() const {
  return str;
}


int main() {
  StringUsed s1 ("test"), s2;
  s2 = s1;            // expected-note{{Calling copy assignment operator for 'StringUsed'}}
                      // expected-note@-1{{Returned allocated memory}}
  s2 = std::move(s1); // expected-note{{Calling move assignment operator for 'StringUsed'}}
  return 0;
}
