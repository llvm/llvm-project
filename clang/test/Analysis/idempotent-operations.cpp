// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-checker=alpha.deadcode.IdempotentOperations -verify %s

// C++ specific false positives

extern void test(int i);
extern void test_ref(int &i);

// Test references affecting pseudoconstants
void false1() {
  int a = 0;
  int five = 5;
  int &b = a;
   test(five * a); // expected-warning {{The right operand to '*' is always 0}}
   b = 4;
}

// Test not flagging idempotent operations because we aborted the analysis
// of a path because of an unsupported construct.
struct RDar9219143_Foo {
  ~RDar9219143_Foo();
  operator bool() const;
};

RDar9219143_Foo foo();
unsigned RDar9219143_bar();
void RDar9219143_test() {
  unsigned i, e;
  for (i = 0, e = RDar9219143_bar(); i != e; ++i)
    if (foo())
      break;  
  if (i == e) // no-warning
    return;
}

