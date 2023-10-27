// RUN: %clang_analyze_cc1  -analyzer-checker=alpha.security.taint,core,alpha.security.ArrayBoundV2 -analyzer-config alpha.security.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config.yaml -Wno-format-security -verify -std=c++11 %s

#define BUFSIZE 10
int Buffer[BUFSIZE];

int scanf(const char*, ...);
int mySource1();
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
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationNamespacePropagation2() {
  int x = mySource3();
  Buffer[x] = 1; // no-warning

  int y = myNamespace::mySource3();
  Buffer[y] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationNamespacePropagation3() {
  int x = myAnotherNamespace::mySource3();
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationNamespacePropagation4() {
  int x;
  // Configured functions without scope should match for all function.
  myNamespace::myScanf("%d", &x);
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationNamespaceFilter1() {
  int x = mySource1();
  if (myNamespace::isOutOfRange2(&x))
    return;
  Buffer[x] = 1; // no-warning

  int y = mySource1();
  if (isOutOfRange2(&y))
    return;
  Buffer[y] = 1; // expected-warning {{Out of bound memory access }}
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
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testReadingFromStdin(char **p) {
  int n;
  fscanf(stdin, "%d", &n);
  Buffer[n] = 1; // expected-warning {{Out of bound memory access (index is tainted)}}
}
