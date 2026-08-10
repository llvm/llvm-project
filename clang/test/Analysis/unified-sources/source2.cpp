// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// This test tests that the warning is here when it is included from
// the unified sources file. The run-line in this file is there
// only to suppress LIT warning for the complete lack of run-line.
int testNullDereference() {
  int *x = 0;
  return *x; // expected-warning{{}}
}

// Let's see if the container inlining heuristic still works.
class ContainerInCodeFile {
  class Iterator {
  };

public:
  Iterator begin() const;
  Iterator end() const;

  int method() { return 0; }
};

int testContainerMethodInCodeFile(ContainerInCodeFile Cont) {
  return 1 / Cont.method(); // expected-warning{{}}
}
