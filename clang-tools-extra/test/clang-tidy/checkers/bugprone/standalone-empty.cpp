// RUN: %check_clang_tidy %s bugprone-standalone-empty %t

namespace std {
template <typename T>
struct vector {
  bool empty();
};

template <typename T>
struct vector_with_clear {
  bool empty();
  void clear();
};

template <typename T>
struct vector_with_void_empty {
  void empty();
  void clear();
};

template <typename T>
struct vector_with_int_empty {
  int empty();
  void clear();
};

template <typename T>
struct vector_with_clear_args {
  bool empty();
  void clear(int i);
};

template <typename T>
struct vector_with_clear_variable {
  bool empty();
  int clear;
};

template <typename T>
bool empty(T &&);

} // namespace std

namespace absl {
struct string {
  bool empty();
};

struct string_with_clear {
  bool empty();
  void clear();
};

struct string_with_void_empty {
  void empty();
  void clear();
};

struct string_with_int_empty {
  int empty();
  void clear();
};

struct string_with_clear_args {
  bool empty();
  void clear(int i);
};

struct string_with_clear_variable {
  bool empty();
  int clear;
};

template <class T>
bool empty(T &&);
} // namespace absl

namespace test {
template <class T>
void empty(T &&);
} // namespace test

namespace base {
template <typename T>
struct base_vector {
    void clear();
};

template <typename T>
struct base_vector_clear_with_args {
    void clear(int i);
};

template <typename T>
struct base_vector_clear_variable {
    int clear;
};

struct base_vector_non_dependent {
    void clear();
};

template <typename T>
struct vector : base_vector<T> {
    bool empty();
};

template <typename T>
struct vector_clear_with_args : base_vector_clear_with_args<T> {
    bool empty();
};

template <typename T>
struct vector_clear_variable : base_vector_clear_variable<T> {
    bool empty();
};

template <typename T>
struct vector_non_dependent : base_vector_non_dependent {
    bool empty();
};

template <typename T>
bool empty(T &&);

} // namespace base

namespace qualifiers {
template <typename T>
struct vector_with_const_clear {
  bool empty() const;
  void clear() const;
};

template <typename T>
struct vector_with_const_empty {
  bool empty() const;
  void clear();
};

template <typename T>
struct vector_with_volatile_clear {
  bool empty() volatile;
  void clear() volatile;
};

template <typename T>
struct vector_with_volatile_empty {
  bool empty() volatile;
  void clear();
};

template <typename T>
bool empty(T &&);
} // namespace qualifiers


void test_member_empty() {
  {
    std::vector<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    std::vector_with_void_empty<int> v;
    v.empty();
    // no-warning
  }

  {
    std::vector_with_clear<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    std::vector_with_int_empty<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    std::vector_with_clear_args<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    std::vector_with_clear_variable<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    absl::string s;
    s.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    absl::string_with_void_empty s;
    s.empty();
    // no-warning
  }

  {
    absl::string_with_clear s;
    s.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }

  {
    absl::string_with_int_empty s;
    s.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }

  {
    absl::string_with_clear_args s;
    s.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    absl::string_with_clear_variable s;
    s.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }
}

void test_qualified_empty() {
  {
    absl::string_with_clear v;
    std::empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}

    absl::empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}

    test::empty(v);
    // no-warning
  }

  {
    absl::string s;
    std::empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    std::empty(0);
    // no-warning
    absl::empty(nullptr);
    // no-warning
  }
}

void test_unqualified_empty() {
  {
    std::vector<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    std::vector_with_void_empty<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    std::vector_with_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    std::vector_with_int_empty<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    std::vector_with_clear_args<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    std::vector_with_clear_variable<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    absl::string s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty' [bugprone-standalone-empty]
  }

  {
    absl::string_with_void_empty s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }

  {
    absl::string_with_clear s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }

  {
    absl::string_with_int_empty s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }

  {
    absl::string_with_clear_args s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty' [bugprone-standalone-empty]
  }

  {
    absl::string_with_clear_variable s;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty' [bugprone-standalone-empty]
  }

  {
    std::vector<int> v;
    using std::empty;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    std::vector_with_clear<int> v;
    using std::empty;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    absl::string s;
    using absl::empty;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty' [bugprone-standalone-empty]
  }

  {
    absl::string_with_clear s;
    using absl::empty;
    empty(s);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'absl::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  s.clear();{{$}}
  }
}

void test_empty_method_expressions() {
  std::vector<int> v;
  bool EmptyReturn(v.empty());
  // no-warning

  (void)v.empty();
  // no-warning

  // Don't warn in the if condition.
  if (v.empty()) v.empty();
  // CHECK-MESSAGES: :[[#@LINE-1]]:18: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]

  // Don't warn in the for condition.
  for(v.empty();v.empty();v.empty()) v.empty();
  // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  // CHECK-MESSAGES: :[[#@LINE-2]]:27: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  // CHECK-MESSAGES: :[[#@LINE-3]]:38: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]

  // Don't warn in the while condition.
  while(v.empty()) v.empty();
  // CHECK-MESSAGES: :[[#@LINE-1]]:20: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]

  // Don't warn in the do-while condition.
  do v.empty(); while(v.empty());
  // CHECK-MESSAGES: :[[#@LINE-1]]:6: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]

  // Don't warn in the switch expression.
  switch(v.empty()) {
    // no-warning
    case true:
      v.empty();
      // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  // Don't warn in the return expression, which is the last statement.
  bool StmtExprReturn = ({v.empty(); v.empty();});
  // CHECK-MESSAGES: :[[#@LINE-1]]:27: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
}

void test_empty_expressions() {
  absl::string s;
  bool test(std::empty(s));
  // no-warning

  (void)std::empty(s);
  // no-warning

  if (std::empty(s)) std::empty(s);
  // CHECK-MESSAGES: :[[#@LINE-1]]:22: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]

  for(std::empty(s);std::empty(s);std::empty(s)) std::empty(s);
  // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  // CHECK-MESSAGES: :[[#@LINE-2]]:35: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  // CHECK-MESSAGES: :[[#@LINE-3]]:50: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]

  while(std::empty(s)) std::empty(s);
  // CHECK-MESSAGES: :[[#@LINE-1]]:24: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]

  do std::empty(s); while(std::empty(s));
  // CHECK-MESSAGES: :[[#@LINE-1]]:6: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]

  switch(std::empty(s)) {
    // no-warning
    case true:
      std::empty(s);
      // CHECK-MESSAGES: :[[#@LINE-1]]:7: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  bool StmtExprReturn = ({std::empty(s); std::empty(s);});
  // CHECK-MESSAGES: :[[#@LINE-1]]:27: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
}

void test_clear_in_base_class() {

  {
    base::vector<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    base::vector_non_dependent<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    base::vector_clear_with_args<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    base::vector_clear_variable<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    base::vector<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'base::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    base::vector_non_dependent<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'base::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    base::vector_clear_with_args<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'base::empty' [bugprone-standalone-empty]
  }

  {
    base::vector_clear_variable<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'base::empty' [bugprone-standalone-empty]
  }
}

void test_clear_with_qualifiers() {
  {
    qualifiers::vector_with_const_clear<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    const qualifiers::vector_with_const_clear<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    const qualifiers::vector_with_const_empty<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    qualifiers::vector_with_const_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'qualifiers::empty' [bugprone-standalone-empty]
  }

  {
    const qualifiers::vector_with_const_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'qualifiers::empty' [bugprone-standalone-empty]
  }

  {
    const std::vector_with_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }

  {
    qualifiers::vector_with_volatile_clear<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    volatile qualifiers::vector_with_volatile_clear<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    volatile qualifiers::vector_with_volatile_empty<int> v;
    v.empty();
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'empty()' [bugprone-standalone-empty]
  }

  {
    qualifiers::vector_with_volatile_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'qualifiers::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    volatile qualifiers::vector_with_volatile_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'qualifiers::empty'; did you mean 'clear()'? [bugprone-standalone-empty]
    // CHECK-FIXES: {{^  }}  v.clear();{{$}}
  }

  {
    volatile std::vector_with_clear<int> v;
    empty(v);
    // CHECK-MESSAGES: :[[#@LINE-1]]:5: warning: ignoring the result of 'std::empty' [bugprone-standalone-empty]
  }
}
