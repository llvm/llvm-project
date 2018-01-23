// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -- -std=c++2a -fno-delayed-template-parsing

struct PositiveBitfieldMember {
  PositiveBitfieldMember() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
  unsigned F : 5;
  // CHECK-FIXES: unsigned F : 5{};
};

struct NegativeUnnamedBitfieldMember {
  NegativeUnnamedBitfieldMember() {}
  unsigned : 5;
};

struct NegativeInitializedBitfieldMembers {
  NegativeInitializedBitfieldMembers() : F(3) { G = 2; }
  unsigned F : 5;
  unsigned G : 5;
};
