// RUN: %check_clang_tidy %s readability-redundant-member-init %t \
// RUN:   -config="{CheckOptions: \
// RUN:             {readability-redundant-member-init.IgnoreMacros: true}}"

struct S {
  S() = default;
  S(int i) : i(i) {}
  S(const char *s) : s(s) {}
  int i = 1;
  const char *s = nullptr;
};

#define BRACES {}
#define EMPTY
#define NUMBER 1
#define NAME

struct WithMacro1 {
  S s BRACES;
};

struct WithMacro2 {
  S s {EMPTY};
};

struct WithMacro3 {
  S s {NUMBER};
};

struct WithMacro4 {
  WithMacro4() : s(EMPTY) {};

  S s;
};

struct WithMacro5 {
  S s{NAME};
};

struct WithoutMacro {
  S s {};
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: initializer for member 's' is redundant
  // CHECK-FIXES: S s;
};

struct WithBlockCommentInInit {
  S s {/* default-construct */};
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: initializer for member 's' is redundant
  // CHECK-FIXES: S s;
};

struct WithCommentInCtorInit {
  WithCommentInCtorInit() : s(/* default */) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: initializer for member 's' is redundant
  // CHECK-FIXES: WithCommentInCtorInit() {};

  S s;
};
struct WithCommentBeforeMacro {
  S s {/* leading */ EMPTY};
};

struct WithCommentAfterMacro {
  S s {EMPTY /* trailing */};
};

struct WithCommentAroundMacroCtor {
  WithCommentAroundMacroCtor() : s(/* begin */ EMPTY /* end */) {};

  S s;
};

struct WithoutMacroCtor {
  WithoutMacroCtor() : s() {};
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: initializer for member 's' is redundant
  // CHECK-FIXES: WithoutMacroCtor() {};

  S s;
};
