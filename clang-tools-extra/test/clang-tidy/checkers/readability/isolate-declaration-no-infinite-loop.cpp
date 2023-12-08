// RUN: %check_clang_tidy -expect-clang-tidy-error %s readability-isolate-declaration %t

int main(){
  int a, b
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple declarations in a single statement reduces readability
  // CHECK-MESSAGES: [[@LINE-2]]:11: error: expected ';' at end of declaration [clang-diagnostic-error]
}
