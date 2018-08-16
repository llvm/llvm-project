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

// RUN: clang-doc --format=md --doxygen --public --extra-arg=-fmodules-ts -p %t %t/test.cpp -output=%t/docs


// RUN: cat %t/docs/./Class.md | FileCheck %s --check-prefix CHECK-0
// CHECK-0: # class Class
// CHECK-0: *Defined at line 32 of test*
// CHECK-0: ## Members
// CHECK-0: int publicField
// CHECK-0: protected int protectedField
// CHECK-0: ## Functions
// CHECK-0: ### void publicMethod()
// CHECK-0: ### void protectedMethod()

// RUN: cat %t/docs/./named.md | FileCheck %s --check-prefix CHECK-1
// CHECK-1: # namespace named
// CHECK-1: ## Functions
// CHECK-1: ### void namedFunction()
// CHECK-1: ### void namedInlineFunction()

// RUN: cat %t/docs/./GlobalNamespace.md | FileCheck %s --check-prefix CHECK-2
// CHECK-2: # Global Namespace
// CHECK-2: ## Functions
// CHECK-2: ### void function(int x)
// CHECK-2: ### int inlinedFunction(int x)
// CHECK-2: ### int functionWithInnerClass(int x)
// CHECK-2: *Defined at line 14 of test*
// CHECK-2: ### int inlinedFunctionWithInnerClass(int x)
// CHECK-2: *Defined at line 23 of test*

// RUN: cat %t/docs/named/NamedClass.md | FileCheck %s --check-prefix CHECK-3
// CHECK-3: # class NamedClass
// CHECK-3: *Defined at line 47 of test*
// CHECK-3: ## Members
// CHECK-3: int namedPublicField
// CHECK-3: protected int namedProtectedField
// CHECK-3: ## Functions
// CHECK-3: ### void namedPublicMethod()
// CHECK-3: ### void namedProtectedMethod()
