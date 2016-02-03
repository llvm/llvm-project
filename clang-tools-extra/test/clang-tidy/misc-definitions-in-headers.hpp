// RUN: %check_clang_tidy %s misc-definitions-in-headers %t

int f() {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'f' defined in a header file; function definitions in header files can lead to ODR violations [misc-definitions-in-headers]
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
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: function 'f3' defined in a header file;
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
// CHECK-MESSAGES: :[[@LINE+1]]:5: warning: function 'f3' defined in a header file;
int f3() {
  int a = 1;
  return a;
}

int f5(); // OK: function declaration.
inline int f6() { return 1; } // OK: inline function definition.
namespace {
  int f7() { return 1; }
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: function 'f7' defined in a header file;
}

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
  int e = 2;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable 'e' defined in a header file;
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
  return 0;
}

template <typename T>
int CD<T, int>::f() { // OK: partial template specialization.
  return 0;
}
