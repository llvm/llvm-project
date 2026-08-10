// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s

class A {
public:
  int method();
};

A *foo();
void bar(A *);

int index;

// We want to check here that the notes about the origins of the null pointer
// (array[index] = foo()) will get to the final report.
//
// The analyzer used to drop exploded nodes for array subscripts when it was
// time to collect redundant nodes. This GC-like mechanism kicks in only when
// the exploded graph is large enough (>1K nodes). For this reason, 'index'
// is a global variable, and the sink point is inside of a loop.

void test() {
  A *array[42];
  A *found;

  for (index = 0; (array[index] = foo()); ++index) { // expected-note {{Loop condition is false. Execution continues on line 34}}
    // expected-note@-1 {{Value assigned to 'index'}}
    // expected-note@-2 {{Assigning value}}
    // expected-note@-3 {{Assuming pointer value is null}}
    if (array[0])
      break;
  }

  do {
    found = array[index]; // expected-note {{Null pointer value stored to 'found'}}

    if (found->method()) // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
      // expected-note@-1 {{Called C++ object pointer is null}}
      bar(found);
  } while (--index);
}
