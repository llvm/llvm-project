// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.ReturnValue \
// RUN:  -analyzer-output=text -verify %s

struct Foo { int Field; };
bool problem();
void doSomething();

// Test the normal case when the implementation of MCAsmParser::Error() (one of
// the methods modeled by this checker) is opaque.
namespace test_normal {
struct MCAsmParser {
  static bool Error();
};

bool parseFoo(Foo &F) {
  if (problem()) {
    // expected-note@-1 {{Assuming the condition is false}}
    // expected-note@-2 {{Taking false branch}}
    return MCAsmParser::Error();
  }

  F.Field = 0;
  // expected-note@-1 {{The value 0 is assigned to 'F.Field'}}
  return false;
}

bool parseFile() {
  Foo F;
  if (parseFoo(F)) {
    // expected-note@-1 {{Calling 'parseFoo'}}
    // expected-note@-2 {{Returning from 'parseFoo'}}
    // expected-note@-3 {{Taking false branch}}
    return true;
  }

  // The following expression would produce the false positive report
  //    "The left operand of '==' is a garbage value"
  // without the modeling done by apiModeling.llvm.ReturnValue:
  if (F.Field == 0) {
    // expected-note@-1 {{Field 'Field' is equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    doSomething();
  }

  // Trigger a zero division to get path notes:
  (void)(1 / F.Field);
  // expected-warning@-1 {{Division by zero}}
  // expected-note@-2 {{Division by zero}}
  return false;
}
} // namespace test_normal


// Sanity check for the highly unlikely case where the implementation of the
// method breaks the convention.
namespace test_break {
struct MCAsmParser {
  static bool Error() {
    return false;
  }
};

bool parseFoo(Foo &F) {
  if (problem()) {
    // expected-note@-1 {{Assuming the condition is false}}
    // expected-note@-2 {{Taking false branch}}
    return !MCAsmParser::Error();
  }

  F.Field = 0;
  // expected-note@-1 {{The value 0 is assigned to 'F.Field'}}
  return MCAsmParser::Error();
  // expected-note@-1 {{'MCAsmParser::Error' returned false, breaking the convention that it always returns true}}
}

bool parseFile() {
  Foo F;
  if (parseFoo(F)) {
    // expected-note@-1 {{Calling 'parseFoo'}}
    // expected-note@-2 {{Returning from 'parseFoo'}}
    // expected-note@-3 {{Taking false branch}}
    return true;
  }

  (void)(1 / F.Field);
  // expected-warning@-1 {{Division by zero}}
  // expected-note@-2 {{Division by zero}}
  return false;
}
} // namespace test_break
