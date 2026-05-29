// RUN: %check_clang_tidy -std=c++26-or-later %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing

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

template <typename T> void constexpr_template_static_assert() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (sizeof(T) == 1) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if constexpr (false) {
      static_assert(false, "discarded in template context");
    }
  }
  // CHECK-FIXES: if constexpr ((sizeof(T) == 1) && (false))
  // CHECK-FIXES: static_assert(false, "discarded in template context");
}
