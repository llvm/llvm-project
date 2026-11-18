// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- -std=c++17 | FileCheck %s

// sysroot(...), header(...) and std(...) are all mandatory and have no default:
// without them nothing is validated, so the typo below is not reported (only
// the pragma itself is diagnosed).
#pragma clang_tidy_analyze_as_header header("<optional>")
// CHECK: :[[@LINE-1]]:9: warning: malformed 'clang_tidy_analyze_as_header'; expected sysroot("...") header("...") std("..."); pragma ignored
#pragma clang_tidy_analyze_as_header sysroot("/x") header("<optional>")
// CHECK: :[[@LINE-1]]:9: warning: malformed 'clang_tidy_analyze_as_header'; expected sysroot("...") header("...") std("..."); pragma ignored

class NotValidated {
public:
  [[clang::analyze_as_method("emplaceTYPO")]] void a();
};
// CHECK-NOT: warning:
