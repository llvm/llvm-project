// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-avoid-capture-default-when-capturing-this %t \
// RUN: -check-suffixes=,DEFAULT
// RUN: %check_clang_tidy -std=c++11-or-later %s cppcoreguidelines-avoid-capture-default-when-capturing-this %t \
// RUN: -config="{CheckOptions: [{key: cppcoreguidelines-avoid-capture-default-when-capturing-this.IgnoreCaptureDefaultByReference, value: true}]}"

struct Obj {
  void lambdas_that_warn_default_capture_copy() {
    int local{};
    int local2{};

    auto explicit_this_capture = [=, this]() { };
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture = [this]() { };

    auto explicit_this_capture_locals1 = [=, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_locals1 = [local, this]() { return (local+x) > 10; };

    auto explicit_this_capture_locals2 = [=, this]() { return (local+local2) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_locals2 = [local, local2, this]() { return (local+local2) > 10; };

    auto explicit_this_capture_local_ref = [=, this, &local]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref = [this, &local]() { return (local+x) > 10; };

    auto explicit_this_capture_local_ref2 = [=, &local, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref2 = [&local, this]() { return (local+x) > 10; };

    auto explicit_this_capture_local_ref3 = [=, &local, this, &local2]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref3 = [&local, this, &local2]() { return (local+x) > 10; };

    auto explicit_this_capture_local_ref4 = [=, &local, &local2, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:46: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref4 = [&local, &local2, this]() { return (local+x) > 10; };

    auto explicit_this_capture_local_ref_extra_whitespace = [=, &  local, &local2, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:62: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref_extra_whitespace = [&  local, &local2, this]() { return (local+x) > 10; };

    auto explicit_this_capture_local_ref_with_comment = [=, & /* byref */ local, &local2, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:58: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto explicit_this_capture_local_ref_with_comment = [& /* byref */ local, &local2, this]() { return (local+x) > 10; };

    auto implicit_this_capture = [=]() { return x > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: lambdas that implicitly capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto implicit_this_capture = [this]() { return x > 10; };

    auto implicit_this_capture_local = [=]() { return (local+x) > 10; };
    // CHECK-MESSAGES: :[[@LINE-1]]:41: warning: lambdas that implicitly capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES: auto implicit_this_capture_local = [local, this]() { return (local+x) > 10; };
  }

  void lambdas_that_warn_default_capture_ref() {
    int local{};
    int local2{};

    auto ref_explicit_this_capture = [&, this]() { };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:39: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES-DEFAULT: auto ref_explicit_this_capture = [this]() { };

    auto ref_explicit_this_capture_local = [&, this]() { return (local+x) > 10; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:45: warning: lambdas that capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES-DEFAULT: auto ref_explicit_this_capture_local = [&local, this]() { return (local+x) > 10; };

    auto ref_implicit_this_capture = [&]() { return x > 10; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:39: warning: lambdas that implicitly capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES-DEFAULT: auto ref_implicit_this_capture = [this]() { return x > 10; };

    auto ref_implicit_this_capture_local = [&]() { return (local+x) > 10; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:45: warning: lambdas that implicitly capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES-DEFAULT: auto ref_implicit_this_capture_local = [&local, this]() { return (local+x) > 10; };

    auto ref_implicit_this_capture_locals = [&]() { return (local+local2+x) > 10; };
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:46: warning: lambdas that implicitly capture 'this' should not specify a capture default [cppcoreguidelines-avoid-capture-default-when-capturing-this]
    // CHECK-FIXES-DEFAULT: auto ref_implicit_this_capture_locals = [&local, &local2, this]() { return (local+local2+x) > 10; };
  }

  void lambdas_that_dont_warn() {
    int local{};
    int local2{};
    auto explicit_this_capture = [this]() { };
    auto explicit_this_capture_local = [this, local]() { return local > 10; };
    auto explicit_this_capture_local_ref = [this, &local]() { return local > 10; };

    auto no_captures = []() {};
    auto capture_local_only = [local]() {};
    auto ref_capture_local_only = [&local]() {};
    auto capture_many = [local, &local2]() {};
  }

  int x;
};
