// RUN: %check_clang_tidy -check-suffixes=ALLOWBOOL -std=c++17-or-later %s readability-redundant-nested-if %t -- -config='{CheckOptions: {readability-redundant-nested-if.AllowUserDefinedBoolConversion: true}}' -- -fno-delayed-template-parsing


bool cond(int X = 0);
int side_effect();
void sink();
void bar();

struct BoolLike {
  operator bool() const;
};

BoolLike make_bool_like();

#define INNER_IF(C) if (C) sink()
#define COND_MACRO cond()
#define OUTER_IF if (cond())

void declaration_condition_boollike_cases() {
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:3: warning: nested 'if' statements can be merged together
  // CHECK-MESSAGES-ALLOWBOOL: :[[@LINE+2]]:5: note: nested 'if' statement to merge declared here
  if (auto Guard = make_bool_like()) {
    if (cond(1))
      sink();
  }
  // CHECK-FIXES-ALLOWBOOL: if (auto Guard = make_bool_like(); static_cast<bool>(Guard) && (cond(1)))
  // CHECK-FIXES-ALLOWBOOL: sink();

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (bool X = COND_MACRO) {
    if (cond(1))
      sink();
  }
}
