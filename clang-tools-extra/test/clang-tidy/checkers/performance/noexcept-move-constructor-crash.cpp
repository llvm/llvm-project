// RUN: %check_clang_tidy -std=c++17 -expect-clang-tidy-error  %s performance-noexcept-move-constructor %t 

template <typename value_type> class set {
  set(set &&) = default;
  set(initializer_list<value_type> __l) {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: error: member 'initializer_list' cannot have template arguments [clang-diagnostic-error]
// CHECK-MESSAGES: :[[@LINE-2]]:36: error: expected ')' [clang-diagnostic-error]
};
