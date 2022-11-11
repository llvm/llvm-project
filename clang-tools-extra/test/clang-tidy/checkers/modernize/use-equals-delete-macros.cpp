// RUN: %check_clang_tidy %s modernize-use-equals-delete %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-equals-delete.IgnoreMacros, value: false}]}"

#define MACRO(type) void operator=(type const &)
class C {
private:
  MACRO(C);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use '= delete' to prohibit calling of a special member function
};
