// RUN: %{clang-tidy} %s --config-file=%{libcxx-dir}/.clang-tidy \
// RUN:                  --load=%{test-tools-dir}/clang_tidy_checks/libcxx-tidy.plugin \
// RUN:                  -checks='-*,libcpp-static-in-constexpr' -- \
// RUN:                  %{flags} %{compile_flags} -Wno-unused-variable | FileCheck %s

void normal_func() {
  static int x = 0;
  thread_local int y = 0;
}

constexpr void constexpr_func() {
  static constexpr int x = 0;
  // CHECK: :[[@LINE-1]]:24: warning: variable of static or thread storage duration inside constexpr function [libcpp-static-in-constexpr]
}

consteval void consteval_func() {
  static constexpr int x = 0;
  // CHECK: :[[@LINE-1]]:24: warning: variable of static or thread storage duration inside constexpr function [libcpp-static-in-constexpr]
}

constexpr void constexpr_with_lambda() {
  auto l = []() {
    static constexpr int x = 0;
    // CHECK: :[[@LINE-1]]:26: warning: variable of static or thread storage duration inside constexpr function [libcpp-static-in-constexpr]
  };
}

constexpr void constexpr_with_constexpr_lambda() {
  auto l = []() constexpr {
    static constexpr int x = 0;
    // CHECK: :[[@LINE-1]]:26: warning: variable of static or thread storage duration inside constexpr function [libcpp-static-in-constexpr]
  };
}

struct S {
    static constexpr void static_member_func() {
        static constexpr int x = 0;
        // CHECK: :[[@LINE-1]]:30: warning: variable of static or thread storage duration inside constexpr function [libcpp-static-in-constexpr]
    }
};
