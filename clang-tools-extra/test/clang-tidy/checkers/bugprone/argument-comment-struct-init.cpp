// RUN: %check_clang_tidy %s bugprone-argument-comment %t

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
  B b2 = {/*z=*/1, /*y=*/2}; // Typo x->z
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
}
