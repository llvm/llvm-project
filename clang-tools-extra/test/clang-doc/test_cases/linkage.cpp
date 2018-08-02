// THIS IS A GENERATED TEST. DO NOT EDIT.
// To regenerate, see clang-doc/gen_test.py docstring.
//
// REQUIRES: system-linux
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "" > %t/compile_flags.txt
// RUN: cp "%s" "%t/test.cpp"

void function(int x);

inline int inlinedFunction(int x);

int functionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

inline int inlinedFunctionWithInnerClass(int x) {
  class InnerClass { //VisibleNoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

class Class {
public:
  void publicMethod();
  int publicField;

protected:
  void protectedMethod();
  int protectedField;

private:
  void privateMethod();
  int privateField;
};

namespace named {
class NamedClass {
public:
  void namedPublicMethod();
  int namedPublicField;

protected:
  void namedProtectedMethod();
  int namedProtectedField;

private:
  void namedPrivateMethod();
  int namedPrivateField;
};

void namedFunction();
static void namedStaticFunction();
inline void namedInlineFunction();
} // namespace named

static void staticFunction(int x); //Internal Linkage

static int staticFunctionWithInnerClass(int x) {
  class InnerClass { //NoLinkage
  public:
    int innerPublicMethod() { return 2; };
  }; //end class
  InnerClass temp;
  return temp.innerPublicMethod();
};

namespace {
class AnonClass {
public:
  void anonPublicMethod();
  int anonPublicField;

protected:
  void anonProtectedMethod();
  int anonProtectedField;

private:
  void anonPrivateMethod();
  int anonPrivateField;
};

void anonFunction();
static void anonStaticFunction();
inline void anonInlineFunction();
} // namespace
