// RUN: %check_clang_tidy -std=c++23-or-later %s readability-redundant-nested-if %t -- -- -fno-delayed-template-parsing

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


void cxx23_positive_anchor(bool B1, bool B2) {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (B1) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: note: nested 'if' statement to merge declared here
    if (B2)
      sink();
  }
  // CHECK-FIXES: if ((B1) && (B2))
  // CHECK-FIXES: sink();
}

void consteval_cases(bool B1, bool B2) {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if consteval {
    if (B1)
      sink();
  } else {
    sink();
  }

  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if (!B1) {
    if consteval {
      sink();
    } else {
      if (B2)
        sink();
    }
  }
}

template <bool B> constexpr void consteval_nested_in_constexpr() {
  // CHECK-MESSAGES-NOT: :[[@LINE+1]]:3: warning: nested 'if' statements can be merged together
  if constexpr (B) {
    if consteval {
      sink();
    }
  }
}
