// RUN: %check_clang_tidy %s misc-definitions-in-headers %t -- --fix-notes

int f() {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f' defined in a header file; function definitions in header files can lead to ODR violations [misc-definitions-in-headers]
// CHECK-MESSAGES: :[[@LINE-2]]:5: note: mark the definition as 'inline'
// CHECK-FIXES: inline int f() {
  return 1;
}

class CA {
  void f1() {} // OK: inline class member function definition.
  void f2();
  template<typename T>
  T f3() {
    T a = 1;
    return a;
  }
  template<typename T>
  struct CAA {
    struct CAB {
      void f4();
    };
  };
};

void CA::f2() { }
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: function 'f2' defined in a header file;
// CHECK-FIXES: inline void CA::f2() {

template <>
int CA::f3() {
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: full function template specialization 'f3<int>' defined in a header file;
// CHECK-FIXES: inline int CA::f3() {
  int a = 1;
  return a;
}

template <typename T>
void CA::CAA<T>::CAB::f4() {
// OK: member function definition of a nested template class in a class.
}

template <typename T>
struct CB {
  void f1();
  struct CCA {
    void f2(T a);
  };
  struct CCB;  // OK: forward declaration.
  static int a; // OK: class static data member declaration.
};

template <typename T>
void CB<T>::f1() { // OK: Member function definition of a class template.
}

template <typename T>
void CB<T>::CCA::f2(T a) {
// OK: member function definition of a nested class in a class template.
}

template <typename T>
struct CB<T>::CCB {
  void f3();
};

template <typename T>
void CB<T>::CCB::f3() {
// OK: member function definition of a nested class in a class template.
}

template <typename T>
int CB<T>::a = 2; // OK: static data member definition of a class template.

template class CB<int>; // OK: explicitly instantiated static data member of a class template.
inline int callCB() {
  CB<double> cb; // OK: implicitly instantiated static data member of a class template.
  return cb.a;
}

template <typename T>
T tf() { // OK: template function definition.
  T a;
  return a;
}


namespace NA {
  int f() { return 1; }
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: function 'f' defined in a header file;
// CHECK-FIXES: inline int f() { return 1; }
}

template <typename T>
T f3() {
  T a = 1;
  return a;
}

template <>
int f3() {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: full function template specialization 'f3<int>' defined in a header file;
// CHECK-FIXES: inline int f3() {
  int a = 1;
  return a;
}

int f5(); // OK: function declaration.
inline int f6() { return 1; } // OK: inline function definition.
namespace {
  int f7() { return 1; }  // OK: each TU defines the function in a unique namespace.
}

int f8() = delete; // OK: the function being marked delete is not callable.

template <typename T>
int f9(T t) { return 1; }

inline void callF9() { f9(1); } // OK: implicitly instantiated function.
template int f9(double); // OK: explicitly instantiated function.

int a = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'a' defined in a header file; variable definitions in header files can lead to ODR violations [misc-definitions-in-headers]
CA a1;
// CHECK-MESSAGES: :[[@LINE-1]]:4: warning: variable 'a1' defined in a header file;

namespace NB {
  int b = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'b' defined in a header file;
  const int c = 1; // OK: internal linkage variable definition.
}

class CC {
  static int d; // OK: class static data member declaration.
};

int CC::d = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'd' defined in a header file;

const char* ca = "foo";
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'ca' defined in a header file;

namespace {
  int e = 2;  // OK: each TU defines the variable in a unique namespace.
}

const char* const g = "foo"; // OK: internal linkage variable definition.
static int h = 1; // OK: internal linkage variable definition.
const int i = 1; // OK: internal linkage variable definition.
extern int j; // OK: internal linkage variable definition.

template <typename T, typename U>
struct CD {
  int f();
};

template <typename T>
struct CD<T, int> {
  int f();
};

template <>
struct CD<int, int> {
  int f();
};

int CD<int, int>::f() {
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: function 'f' defined in a header file;
// CHECK-FIXES: inline int CD<int, int>::f() {
  return 0;
}

template <typename T>
int CD<T, int>::f() { // OK: partial template specialization.
  return 0;
}

constexpr int k = 1; // OK: constexpr variable has internal linkage.

constexpr int f10() { return 0; } // OK: constexpr function definition.

const int f11() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: function 'f11' defined in a header file;
// CHECK-FIXES: inline const int f11() { return 0; }

template <typename T>
const T f12();

template <>
const int f12() { return 0; }
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: full function template specialization 'f12<int>' defined in a header file;
// CHECK-FIXES: inline const int f12() { return 0; }

template <typename T1, typename T2>
constexpr bool f13 = false;

template <typename T>
constexpr bool f13<T, int> = true; // OK: template partial specialization

template <>
constexpr bool f13<void, int> = false;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: variable 'f13<void, int>' defined in a header file;

int main() {}
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'main' defined in a header file;
// CHECK-FIXES: {{^}}int main() {
