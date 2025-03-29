// RUN: %clang %s -std=c++17 -Xclang -verify --analyze \
// RUN:   -Xclang -analyzer-checker=core \
// RUN:   -Xclang -analyzer-checker=debug.ExprInspection \
// RUN:   -Xclang -analyzer-checker=core,alpha.core.StdVariant

#include "Inputs/system-header-simulator-cxx.h"

class Foo{};

void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

//helper functions
void changeVariantType(std::variant<int, char> &v) {
  v = 25;
}

void changesToInt(std::variant<int, char> &v);
void changesToInt(std::variant<int, char> *v);

void cannotChangePtr(const std::variant<int, char> &v);
void cannotChangePtr(const std::variant<int, char> *v);

char getUnknownChar();

void swap(std::variant<int, char> &v1, std::variant<int, char> &v2) {
  std::variant<int, char> tmp = v1;
  v1 = v2;
  v2 = tmp;
}

void cantDo(const std::variant<int, char>& v) {
  std::variant<int, char> vtmp = v;
  vtmp = 5;
  int a = std::get<int> (vtmp);
  (void) a;
}

void changeVariantPtr(std::variant<int, char> *v) {
  *v = 'c';
}

using var_t = std::variant<int, char>;
using var_tt = var_t;
using int_t = int;
using char_t = char;

// A quick sanity check to see that std::variant's std::get
// is not being confused with std::pairs std::get.
void wontConfuseStdGets() {
  std::pair<int, char> p{15, '1'};
  int a = std::get<int>(p);
  char c = std::get<char>(p);
  (void)a;
  (void)c;
}

//----------------------------------------------------------------------------//
// std::get
//----------------------------------------------------------------------------//
void stdGetType() {
  std::variant<int, char> v = 25;
  int a = std::get<int>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void stdGetPointer() {
  int *p = new int;
  std::variant<int*, char> v = p;
  int *a = std::get<int*>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int *', not a 'char'}}
  (void)a;
  (void)c;
  delete p;
}

void stdGetObject() {
  std::variant<int, char, Foo> v = Foo{};
  Foo f = std::get<Foo>(v);
  int i = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'Foo', not an 'int'}}
  (void)i;
}

void stdGetPointerAndPointee() {
  int a = 5;
  std::variant<int, int*> v = &a;
  int *b = std::get<int*>(v);
  int c = std::get<int>(v); // expected-warning {{std::variant 'v' held an 'int *', not an 'int'}}
  (void)c;
  (void)b;
}

void variantHoldingVariant() {
  std::variant<std::variant<int, char>, std::variant<char, int>> v = std::variant<int,char>(25);
  std::variant<int, char> v1 = std::get<std::variant<int,char>>(v);
  std::variant<char, int> v2 = std::get<std::variant<char,int>>(v); // expected-warning {{std::variant 'v' held a 'std::variant<int, char>', not a 'class std::variant<char, int>'}}
}

//----------------------------------------------------------------------------//
// Constructors and assignments
//----------------------------------------------------------------------------//
void copyConstructor() {
  std::variant<int, char> v = 25;
  std::variant<int, char> t(v);
  int a = std::get<int> (t);
  char c = std::get<char> (t); // expected-warning {{std::variant 't' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void copyAssignmentOperator() {
  std::variant<int, char> v = 25;
  std::variant<int, char> t = 'c';
  t = v;
  int a = std::get<int> (t);
  char c = std::get<char> (t); // expected-warning {{std::variant 't' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void assignmentOperator() {
  std::variant<int, char> v = 25;
  int a = std::get<int> (v);
  (void)a;
  v = 'c';
  char c = std::get<char>(v);
  a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void typeChangeThreeTimes() {
  std::variant<int, char, float> v = 25;
  int a = std::get<int> (v);
  (void)a;
  v = 'c';
  char c = std::get<char>(v);
  v = 25;
  a = std::get<int>(v);
  (void)a;
  v = 1.25f;
  float f = std::get<float>(v);
  a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'float', not an 'int'}}
  (void)a;
  (void)c;
  (void)f;
}

void defaultConstructor() {
  std::variant<int, char> v;
  int i = std::get<int>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)i;
  (void)c;
}

// Verify that we handle temporary objects correctly
void temporaryObjectsConstructor() {
  std::variant<int, char> v(std::variant<int, char>('c'));
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void temporaryObjectsAssignment() {
  std::variant<int, char> v = std::variant<int, char>('c');
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

// Verify that we handle pointer types correctly
void pointerTypeHeld() {
  int *p = new int;
  std::variant<int*, char> v = p;
  int *a = std::get<int*>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int *', not a 'char'}}
  (void)a;
  (void)c;
  delete p;
}

std::variant<int, char> get_unknown_variant();
// Verify that the copy constructor is handles properly when the std::variant
// has no previously activated type and we copy an object of unknown value in it.
void copyFromUnknownVariant() {
  std::variant<int, char> u = get_unknown_variant();
  std::variant<int, char> v(u);
  int a = std::get<int>(v); // no-waring
  char c = std::get<char>(v); // no-warning
  (void)a;
  (void)c;
}

// Verify that the copy constructor is handles properly when the std::variant
// has previously activated type and we copy an object of unknown value in it.
void copyFromUnknownVariantBef() {
  std::variant<int, char> v = 25;
  std::variant<int, char> u = get_unknown_variant();
  v = u;
  int a = std::get<int>(v); // no-waring
  char c = std::get<char>(v); // no-warning
  (void)a;
  (void)c;
}

//----------------------------------------------------------------------------//
// typedef
//----------------------------------------------------------------------------//

void typefdefedVariant() {
  var_t v = 25;
  int a = std::get<int>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void typedefedTypedfefedVariant() {
  var_tt v = 25;
  int a = std::get<int>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void typedefedGet() {
  std::variant<char, int> v = 25;
  int a = std::get<int_t>(v);
  char c = std::get<char_t>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void typedefedPack() {
  std::variant<int_t, char_t> v = 25;
  int a = std::get<int>(v);
  char c = std::get<char>(v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

void fromVariable() {
  char o = 'c';
  std::variant<int, char> v(o);
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void unknowValueButKnownType() {
  char o = getUnknownChar();
  std::variant<int, char> v(o);
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void createPointer() {
  std::variant<int, char> *v = new std::variant<int, char>(15);
  int a = std::get<int>(*v);
  char c = std::get<char>(*v); // expected-warning {{std::variant  held an 'int', not a 'char'}}
  (void)a;
  (void)c;
  delete v;
}

//----------------------------------------------------------------------------//
// Passing std::variants to functions
//----------------------------------------------------------------------------//

// Verifying that we are not invalidating the memory region of a variant if
// a non inlined or inlined function takes it as a constant reference or pointer
void constNonInlineRef() {
  std::variant<int, char> v = 'c';
  cannotChangePtr(v);
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void contNonInlinePtr() {
  std::variant<int, char> v = 'c';
  cannotChangePtr(&v);
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void copyInAFunction() {
  std::variant<int, char> v = 'c';
  cantDo(v);
  char c = std::get<char>(v);
  int a = std::get<int>(v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;

}

// Verifying that we can keep track of the type stored in std::variant when
// it is passed to an inlined function as a reference or pointer
void changeThruPointers() {
  std::variant<int, char> v = 15;
  changeVariantPtr(&v);
  char c = std::get<char> (v);
  int a = std::get<int> (v); // expected-warning {{std::variant 'v' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void functionCallWithCopyAssignment() {
  var_t v1 = 15;
  var_t v2 = 'c';
  swap(v1, v2);
  int a = std::get<int> (v2);
  (void)a;
  char c = std::get<char> (v1);
  a = std::get<int> (v1); // expected-warning {{std::variant 'v1' held a 'char', not an 'int'}}
  (void)a;
  (void)c;
}

void inlineFunctionCall() {
  std::variant<int, char> v = 'c';
  changeVariantType(v);
  int a = std::get<int> (v);
  char c = std::get<char> (v); // expected-warning {{std::variant 'v' held an 'int', not a 'char'}}
  (void)a;
  (void)c;
}

// Verifying that we invalidate the mem region of std::variant when it is
// passed as a non const reference or a pointer to a non inlined function.
void nonInlineFunctionCall() {
  std::variant<int, char> v = 'c';
  changesToInt(v);
  int a = std::get<int> (v); // no-waring
  char c = std::get<char> (v); // no-warning
  (void)a;
  (void)c;
}

void nonInlineFunctionCallPtr() {
  std::variant<int, char> v = 'c';
  changesToInt(&v);
  int a = std::get<int> (v); // no-warning
  char c = std::get<char> (v); // no-warning
  (void)a;
  (void)c;
}