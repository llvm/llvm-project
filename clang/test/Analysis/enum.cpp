// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=debug.ExprInspection %s

void clang_analyzer_eval(bool);

enum class Foo {
  Zero
};

bool pr15703(int x) {
  return Foo::Zero == (Foo)x; // don't crash
}

void testCasting(int i) {
  Foo f = static_cast<Foo>(i);
  int j = static_cast<int>(f);
  if (i == 0)
  {
    clang_analyzer_eval(f == Foo::Zero); // expected-warning{{TRUE}}
    clang_analyzer_eval(j == 0); // expected-warning{{TRUE}}
  }
  else
  {
    clang_analyzer_eval(f == Foo::Zero); // expected-warning{{FALSE}}
    clang_analyzer_eval(j == 0); // expected-warning{{FALSE}}
  }
}

enum class EnumBool : bool {
  F = false,
  T = true
};

bool testNoCrashOnSwitchEnumBool(EnumBool E) {
  switch (E) {
  case EnumBool::F:
    return false;
  }
  return true;
}

bool testNoCrashOnSwitchEnumBoolConstant() {
  EnumBool E = EnumBool::F;
  switch (E) {
    case EnumBool::F:
      return false; 
  }
  return true; 
}

typedef __INTPTR_TYPE__ intptr_t;
bool testNoCrashOnSwitchEnumBoolConstantCastedFromNullptr() {
  EnumBool E = static_cast<EnumBool>((intptr_t)nullptr);
  switch (E) {
  case EnumBool::F:
    return false;
  }
  return true;
}

bool testNoCrashOnSwitchEnumBoolConstantCastedFromPtr() {
  int X;
  intptr_t P = (intptr_t)&X;
  EnumBool E = static_cast<EnumBool>(P);
  switch (E) {
  case EnumBool::F:
    return false;
  }
  return true;
}
