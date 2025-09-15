// RUN: %clang_analyze_cc1 -std=c++11 -Wno-format-security \
// RUN:   -analyzer-checker=core,optin.taint,security.ArrayBound,debug.ExprInspection \
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

  // It's fine to not propagate taint to empty classes, since they don't have any data members.
  Empty E = mySource1<Empty>();
  clang_analyzer_isTainted(E); // expected-warning {{NO}}

  Aggr A = mySource1<Aggr>();
  clang_analyzer_isTainted(A);      // expected-warning {{YES}}
  clang_analyzer_isTainted(A.data); // expected-warning {{YES}}
}
} // namespace gh114270


namespace format_attribute {
__attribute__((__format__ (__printf__, 1, 2)))
void log_freefunc(const char *fmt, ...);

void test_format_attribute_freefunc() {
  int n;
  fscanf(stdin, "%d", &n); // Get a tainted value.
                           
  log_freefunc("This number is suspicious: %d\n", n); // no-warning
}

struct Foo {
  // When the format attribute is applied to a method, argumet '1' is the
  // implicit `this`, so e.g. in this case argument '2' specifies `fmt`.
  // Specifying '1' instead of '2' would produce a compilation error:
  // "format attribute cannot specify the implicit this argument as the format string"
  __attribute__((__format__ (__printf__, 2, 3)))
  void log_method(const char *fmt, ...);

  void test_format_attribute_method() {
    int n;
    fscanf(stdin, "%d", &n); // Get a tainted value.
                             
    // The analyzer used to misinterpret the parameter indices in the format
    // attribute when the format attribute is applied to a method.
    log_method("This number is suspicious: %d\n", n); // no-warning
  }

  __attribute__((__format__ (__printf__, 1, 2)))
  static void log_static_method(const char *fmt, ...);

  void test_format_attribute_static_method() {
    int n;
    fscanf(stdin, "%d", &n); // Get a tainted value.
                             
    log_static_method("This number is suspicious: %d\n", n); // no-warning
  }
};

} // namespace format_attribute
