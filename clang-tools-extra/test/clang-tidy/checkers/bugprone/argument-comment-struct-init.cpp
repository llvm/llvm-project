// RUN: %check_clang_tidy -std=c++17 %s bugprone-argument-comment %t

struct A {
  int x, y;
};

void testA() {
  A a1 = {/*x=*/1, /*y=*/2};
  A a2 = {/*y=*/1, /*x=*/2};
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: argument name 'y' in comment does not match parameter name 'x' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:20: warning: argument name 'x' in comment does not match parameter name 'y' [bugprone-argument-comment]

  A a3 = {
    /*x=*/1,
    /*y=*/2
  };

  A a4 = {
    /*y=*/1,
    /*x=*/2
  };
  // CHECK-MESSAGES: [[@LINE-3]]:5: warning: argument name 'y' in comment does not match parameter name 'x' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-3]]:5: warning: argument name 'x' in comment does not match parameter name 'y' [bugprone-argument-comment]
}

struct B {
  int x, y, z;
};

void testB() {
  B b1 = {/*x=*/1, /*y=*/2}; // Partial init
  B b2 = {/*z=*/1, /*y=*/2};
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: argument name 'z' in comment does not match parameter name 'x' [bugprone-argument-comment]
}

struct BitFields {
  int a : 4;
  int : 4;
  int b : 4;
};

void testBitFields() {
  BitFields b1 = {/*a=*/1, /*b=*/2};
  BitFields b2 = {/*a=*/1, /*c=*/2};
  // CHECK-MESSAGES: [[@LINE-1]]:28: warning: argument name 'c' in comment does not match parameter name 'b' [bugprone-argument-comment]
}

struct CorrectFix {
  int long_field_name;
  int other;
};

void testFix() {
  CorrectFix c = {/*long_feild_name=*/1, 2};
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: argument name 'long_feild_name' in comment does not match parameter name 'long_field_name' [bugprone-argument-comment]
  // CHECK-FIXES: CorrectFix c = {/*long_field_name=*/1, 2};
}

struct Base {
  int b;
};

struct Derived : Base {
  int d;
};

void testInheritance() {
  Derived d1 = {/*b=*/ 1, /*d=*/ 2};
  Derived d2 = {/*x=*/ 1, /*d=*/ 2};
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: argument name 'x' in comment does not match parameter name 'b' [bugprone-argument-comment]
  Derived d3 = {/*b=*/ 1, /*x=*/ 2};
  // CHECK-MESSAGES: [[@LINE-1]]:27: warning: argument name 'x' in comment does not match parameter name 'd' [bugprone-argument-comment]
}


struct DerivedExplicit : Base {
  int d;
};

void testInheritanceExplicit() {
  DerivedExplicit d1 = {{/*b=*/ 1}, /*d=*/ 2};
  DerivedExplicit d2 = {{/*x=*/ 1}, /*d=*/ 2};
  // CHECK-MESSAGES: [[@LINE-1]]:26: warning: argument name 'x' in comment does not match parameter name 'b' [bugprone-argument-comment]
}

struct DeepBase {
  int db;
};

struct Middle : DeepBase {
  int m;
};

struct DerivedDeep : Middle {
  int d;
};

void testDeepInheritance() {
  DerivedDeep d1 = {/*db=*/ 1, /*m=*/ 2, /*d=*/ 3};
  DerivedDeep d2 = {/*x=*/ 1, /*m=*/ 2, /*d=*/ 3};
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: argument name 'x' in comment does not match parameter name 'db' [bugprone-argument-comment]
}

struct Inner {
  int i;
};

struct Outer {
  Inner in;
};

void testNestedStruct() {
  Outer o = {/*i=*/1};
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: argument name 'i' in comment does not match parameter name 'in' [bugprone-argument-comment]
}

#define MACRO_VAL 1
void testMacroVal() {
  A a = {/*x=*/ MACRO_VAL, /*y=*/ 2};
  A a2 = {/*y=*/ MACRO_VAL, /*x=*/ 2};
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: argument name 'y' in comment does not match parameter name 'x' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:29: warning: argument name 'x' in comment does not match parameter name 'y' [bugprone-argument-comment]
}

#define MACRO_INIT { /*x=*/ 1, /*y=*/ 2 }
#define MACRO_INIT_BAD { /*y=*/ 1, /*x=*/ 2 }

void testMacroInit() {
  A a = MACRO_INIT;
  A a2 = MACRO_INIT_BAD;
  // Won't flag warnings.
}
