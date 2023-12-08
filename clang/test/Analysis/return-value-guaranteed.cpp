// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core,apiModeling.llvm.ReturnValue \
// RUN:  -analyzer-output=text -verify=class %s

struct Foo { int Field; };
bool problem();
void doSomething();

// We predefined the return value of 'MCAsmParser::Error' as true and we cannot
// take the false-branches which leads to a "garbage value" false positive.
namespace test_classes {
struct MCAsmParser {
  static bool Error();
};

bool parseFoo(Foo &F) {
  if (problem()) {
    // class-note@-1 {{Assuming the condition is false}}
    // class-note@-2 {{Taking false branch}}
    return MCAsmParser::Error();
  }

  F.Field = 0;
  // class-note@-1 {{The value 0 is assigned to 'F.Field'}}
  return !MCAsmParser::Error();
  // class-note@-1 {{'MCAsmParser::Error' returns true}}
}

bool parseFile() {
  Foo F;
  if (parseFoo(F)) {
    // class-note@-1 {{Calling 'parseFoo'}}
    // class-note@-2 {{Returning from 'parseFoo'}}
    // class-note@-3 {{Taking false branch}}
    return true;
  }

  if (F.Field == 0) {
    // class-note@-1 {{Field 'Field' is equal to 0}}
    // class-note@-2 {{Taking true branch}}

    // no-warning: "The left operand of '==' is a garbage value" was here.
    doSomething();
  }

  (void)(1 / F.Field);
  // class-warning@-1 {{Division by zero}}
  // class-note@-2 {{Division by zero}}
  return false;
}
} // namespace test_classes


// We predefined 'MCAsmParser::Error' as returning true, but now it returns
// false, which breaks our invariant. Test the notes.
namespace test_break {
struct MCAsmParser {
  static bool Error() {
    return false; // class-note {{'MCAsmParser::Error' returns false}}
  }
};

bool parseFoo(Foo &F) {
  if (problem()) {
    // class-note@-1 {{Assuming the condition is false}}
    // class-note@-2 {{Taking false branch}}
    return !MCAsmParser::Error();
  }

  F.Field = 0;
  // class-note@-1 {{The value 0 is assigned to 'F.Field'}}
  return MCAsmParser::Error();
  // class-note@-1 {{Calling 'MCAsmParser::Error'}}
  // class-note@-2 {{Returning from 'MCAsmParser::Error'}}
}

bool parseFile() {
  Foo F;
  if (parseFoo(F)) {
    // class-note@-1 {{Calling 'parseFoo'}}
    // class-note@-2 {{Returning from 'parseFoo'}}
    // class-note@-3 {{Taking false branch}}
    return true;
  }

  (void)(1 / F.Field);
  // class-warning@-1 {{Division by zero}}
  // class-note@-2 {{Division by zero}}
  return false;
}
} // namespace test_classes
