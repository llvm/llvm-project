#ifndef NON_VISIBLE_REFERENCES_H
#define NON_VISIBLE_REFERENCES_H

using int32_t = int;

namespace HeaderValues {
constexpr int Constant = 5;
static int Static = 6;
enum Enum { EnumValue = 7 };
} // namespace HeaderValues

#define NON_VISIBLE_INIT_VALUE CPP_MACRO_CONSTANT

class NonVisibleConstexpr {
public:
  NonVisibleConstexpr();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class NonVisibleStatic {
public:
  NonVisibleStatic();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class NonVisibleEnum {
public:
  NonVisibleEnum();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class NonVisibleNestedCast {
public:
  NonVisibleNestedCast();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class NonVisibleMacroReference {
public:
  NonVisibleMacroReference();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class NonVisibleUsingDeclaration {
public:
  NonVisibleUsingDeclaration();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

template <typename T>
class NonVisibleTemplate {
public:
  NonVisibleTemplate();

private:
  int32_t member;
  // CHECK-MESSAGES-ALLOW: :[[@LINE-1]]:11: warning: use default member initializer for 'member'
};

class HeaderVisibleConstexpr {
public:
  HeaderVisibleConstexpr();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{HeaderValues::Constant};
  // CHECK-FIXES-ALLOW: int member{HeaderValues::Constant};
};

class HeaderVisibleStatic {
public:
  HeaderVisibleStatic();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{HeaderValues::Static};
  // CHECK-FIXES-ALLOW: int member{HeaderValues::Static};
};

class HeaderVisibleEnum {
public:
  HeaderVisibleEnum();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{HeaderValues::EnumValue};
  // CHECK-FIXES-ALLOW: int member{HeaderValues::EnumValue};
};

class SameClassLaterStatic {
public:
  SameClassLaterStatic();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{ClassConstant};
  // CHECK-FIXES-ALLOW: int member{ClassConstant};
  static constexpr int ClassConstant = 8;
};

class SameClassLaterEnum {
public:
  SameClassLaterEnum();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{ClassEnumValue};
  // CHECK-FIXES-ALLOW: int member{ClassEnumValue};
  enum { ClassEnumValue = 9 };
};

struct BaseWithStatic {
  static constexpr int BaseConstant = 10;
};

class InheritedStatic : public BaseWithStatic {
public:
  InheritedStatic();

private:
  int member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use default member initializer for 'member'
  // CHECK-MESSAGES-ALLOW: :[[@LINE-2]]:7: warning: use default member initializer for 'member'
  // CHECK-FIXES: int member{BaseConstant};
  // CHECK-FIXES-ALLOW: int member{BaseConstant};
};

#endif // NON_VISIBLE_REFERENCES_H
