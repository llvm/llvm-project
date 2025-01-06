// RUN: %clang_analyze_cc1 -std=c++11 -Wno-format-security \
// RUN:   -analyzer-checker=core,optin.taint,alpha.security.ArrayBoundV2,debug.ExprInspection \
// RUN:   -analyzer-config optin.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config.yaml \
// RUN:   -verify %s

template <typename T> void clang_analyzer_isTainted(T);

#define BUFSIZE 10
int Buffer[BUFSIZE];

int scanf(const char*, ...);
template <typename T = int> T mySource1();
int mySource3();

typedef struct _FILE FILE;
extern "C" {
extern FILE *stdin;
}
int fscanf(FILE *stream, const char *format, ...);

bool isOutOfRange2(const int*);

void mySink2(int);

// Test configuration
namespace myNamespace {
  void scanf(const char*, ...);
  void myScanf(const char*, ...);
  int mySource3();

  bool isOutOfRange(const int*);
  bool isOutOfRange2(const int*);

  void mySink(int, int, int);
  void mySink2(int);
}

namespace myAnotherNamespace {
  int mySource3();

  bool isOutOfRange2(const int*);

  void mySink2(int);
}

void testConfigurationNamespacePropagation1() {
  int x;
  // The built-in functions should be matched only for functions in
  // the global namespace
  myNamespace::scanf("%d", &x);
  Buffer[x] = 1; // no-warning

  scanf("%d", &x);
  Buffer[x] = 1; // expected-warning {{Potential out of bound access }}
}

void testConfigurationNamespacePropagation2() {
  int x = mySource3();
  Buffer[x] = 1; // no-warning

  int y = myNamespace::mySource3();
  Buffer[y] = 1; // expected-warning {{Potential out of bound access }}
}

void testConfigurationNamespacePropagation3() {
  int x = myAnotherNamespace::mySource3();
  Buffer[x] = 1; // expected-warning {{Potential out of bound access }}
}

void testConfigurationNamespacePropagation4() {
  int x;
  // Configured functions without scope should match for all function.
  myNamespace::myScanf("%d", &x);
  Buffer[x] = 1; // expected-warning {{Potential out of bound access }}
}

void testConfigurationNamespaceFilter1() {
  int x = mySource1();
  if (myNamespace::isOutOfRange2(&x))
    return;
  Buffer[x] = 1; // no-warning

  int y = mySource1();
  if (isOutOfRange2(&y))
    return;
  Buffer[y] = 1; // expected-warning {{Potential out of bound access }}
}

void testConfigurationNamespaceFilter2() {
  int x = mySource1();
  if (myAnotherNamespace::isOutOfRange2(&x))
    return;
  Buffer[x] = 1; // no-warning
}

void testConfigurationNamespaceFilter3() {
  int x = mySource1();
  if (myNamespace::isOutOfRange(&x))
    return;
  Buffer[x] = 1; // no-warning
}

void testConfigurationNamespaceSink1() {
  int x = mySource1();
  mySink2(x); // no-warning

  int y = mySource1();
  myNamespace::mySink2(y);
  // expected-warning@-1 {{Untrusted data is passed to a user-defined sink}}
}

void testConfigurationNamespaceSink2() {
  int x = mySource1();
  myAnotherNamespace::mySink2(x);
  // expected-warning@-1 {{Untrusted data is passed to a user-defined sink}}
}

void testConfigurationNamespaceSink3() {
  int x = mySource1();
  myNamespace::mySink(x, 0, 1);
  // expected-warning@-1 {{Untrusted data is passed to a user-defined sink}}
}

struct Foo {
    void scanf(const char*, int*);
    void myMemberScanf(const char*, int*);
};

void testConfigurationMemberFunc() {
  int x;
  Foo foo;
  foo.scanf("%d", &x);
  Buffer[x] = 1; // no-warning

  foo.myMemberScanf("%d", &x);
  Buffer[x] = 1; // expected-warning {{Potential out of bound access }}
}

void testReadingFromStdin(char **p) {
  int n;
  fscanf(stdin, "%d", &n);
  Buffer[n] = 1; // expected-warning {{Potential out of bound access }}
}

namespace gh114270 {
class Empty {};
class Aggr {
public:
  int data;
};

void top() {
  int Int = mySource1<int>();
  clang_analyzer_isTainted(Int); // expected-warning {{YES}}

  Empty E = mySource1<Empty>();
  clang_analyzer_isTainted(E); // expected-warning {{YES}}

  Aggr A = mySource1<Aggr>();
  clang_analyzer_isTainted(A);      // expected-warning {{YES}}
  clang_analyzer_isTainted(A.data); // expected-warning {{YES}}
}
} // namespace gh114270
