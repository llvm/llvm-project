// RUN: %check_clang_tidy %s readability-redundant-access-specifiers %t -- \
// RUN:   -config="{CheckOptions: {readability-redundant-access-specifiers.CheckFirstDeclaration: true}}" --

class FooPublic {
private: // comment-0
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the implicit access specifier [readability-redundant-access-specifiers]
  // CHECK-FIXES: {{^}}// comment-0{{$}}
  int a;
};

struct StructPublic {
public: // comment-1
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the implicit access specifier [readability-redundant-access-specifiers]
  // CHECK-FIXES: {{^}}// comment-1{{$}}
  int a;
};

union UnionPublic {
public: // comment-2
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: redundant access specifier has the same accessibility as the implicit access specifier [readability-redundant-access-specifiers]
  // CHECK-FIXES: {{^}}// comment-2{{$}}
  int a;
};

class FooMacro {
#if defined(ZZ)
private:
#endif
  int a;
};

class ValidInnerStruct {
  struct Inner {
  private:
    int b;
  };
};

#define MIXIN private: int b;

class ValidMacro {
  MIXIN
};
