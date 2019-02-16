// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core,cplusplus,unix.Malloc,debug.ExprInspection %s -verify -analyzer-output=text

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

StringUsed& StringUsed::operator=(const StringUsed &rhs) { // expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}} expected-note{{TRUE}} expected-note{{UNKNOWN}}
  free(str); // expected-note{{Memory is released}}
  str = strdup(rhs.str); // expected-warning{{Use of memory after it is freed}}  expected-note{{Use of memory after it is freed}}
  return *this;
}

StringUsed& StringUsed::operator=(StringUsed &&rhs) { // expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}} expected-note{{TRUE}} expected-note{{UNKNOWN}}
  str = rhs.str;
  rhs.str = nullptr; // FIXME: An improved leak checker should warn here
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

StringUnused& StringUnused::operator=(const StringUnused &rhs) { // expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}} expected-note{{TRUE}} expected-note{{UNKNOWN}}
  free(str); // expected-note{{Memory is released}}
  str = strdup(rhs.str); // expected-warning{{Use of memory after it is freed}}  expected-note{{Use of memory after it is freed}}
  return *this;
}

StringUnused& StringUnused::operator=(StringUnused &&rhs) { // expected-note{{Assuming rhs == *this}} expected-note{{Assuming rhs != *this}}
  clang_analyzer_eval(*this == rhs); // expected-warning{{TRUE}} expected-warning{{UNKNOWN}} expected-note{{TRUE}} expected-note{{UNKNOWN}}
  str = rhs.str;
  rhs.str = nullptr; // FIXME: An improved leak checker should warn here
  return *this;
}

StringUnused::operator const char*() const {
  return str;
}


int main() {
  StringUsed s1 ("test"), s2;
  s2 = s1;
  s2 = std::move(s1);
  return 0;
}
