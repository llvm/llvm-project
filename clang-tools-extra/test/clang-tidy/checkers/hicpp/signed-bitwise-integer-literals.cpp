// RUN: %check_clang_tidy %s hicpp-signed-bitwise %t -- \
// RUN:   -config="{CheckOptions: {hicpp-signed-bitwise.IgnorePositiveIntegerLiterals: true}}" \
// RUN: -- -std=c++11

void examples() {
  unsigned UValue = 40u;
  unsigned URes;

  URes = UValue & 1u; //Ok
  URes = UValue & -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use of a signed integer operand with a binary bitwise operator

  unsigned URes2 = URes << 1; //Ok

  int IResult;
  IResult = 10 & 2; //Ok
  IResult = 3 << -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use of a signed integer operand with a binary bitwise operator

  int Int = 30;
  IResult = Int << 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  IResult = ~0; //Ok
}

enum EnumConstruction {
  one = 1,
  two = 2,
  test1 = 1 << 12, //Ok
  test2 = one << two,
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
  test3 = 1u << 12, //Ok
};
