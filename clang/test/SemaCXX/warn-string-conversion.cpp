// RUN: %clang_cc1 -fsyntax-only -Wstring-conversion -verify %s

// Warn on cases where a string literal is converted into a bool.
// An exception is made for this in logical operators.
void assert(bool condition);
void test0() {
  bool b0 = "hi"; // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  b0 = ""; // expected-warning{{implicit conversion turns string literal into bool: 'const char [1]' to 'bool'}}
  b0 = 0 && "";
  assert("error"); // expected-warning{{implicit conversion turns string literal into bool: 'const char [6]' to 'bool'}}
  assert(0 && "error");

  while("hi") {} // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  do {} while("hi"); // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  for (;"hi";); // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  if("hi") {} // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
}

