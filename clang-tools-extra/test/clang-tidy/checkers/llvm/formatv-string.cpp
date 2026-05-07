// RUN: %check_clang_tidy %s llvm-formatv-string %t

namespace llvm {

template <typename... Ts>
void formatv(const char *Fmt, Ts &&...Vals) {}

template <typename... Ts>
void formatv(bool Validate, const char *Fmt, Ts &&...Vals) {}

} // namespace llvm

void correct() {
  llvm::formatv("{0}", 1);
  llvm::formatv("{0} {1}", 1, 2);
  llvm::formatv("{0} {0}", 1);
  llvm::formatv("{1} {0}", 1, 2);
  llvm::formatv("{0,10}", 1);
  llvm::formatv("{0,-10}", 1);
  llvm::formatv("{0:x}", 1);
  llvm::formatv("{0,10:x}", 1);
  llvm::formatv("no replacements");
  llvm::formatv("escaped {{ braces }}");
  llvm::formatv("{}", 1);
  llvm::formatv("{} {}", 1, 2);
  llvm::formatv(false, "{0}", 1);
}

void too_few_args() {
  llvm::formatv("{0} {1}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: formatv() format string requires 2 arguments, but 1 argument was provided

  llvm::formatv("{0} {1} {2}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: formatv() format string requires 3 arguments, but 2 arguments were provided
}

void too_many_args() {
  llvm::formatv("{0}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: formatv() format string requires 1 argument, but 2 arguments were provided

  llvm::formatv("no replacements", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: formatv() format string requires 0 arguments, but 1 argument was provided
}

void mixed_indices() {
  llvm::formatv("{} {1}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: formatv() format string mixes automatic and explicit indices
}

void holes_in_indices() {
  llvm::formatv("{0} {2}", 1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: formatv() argument unused in format string
}

void non_literal_format_string(const char *fmt) {
  // No warning for non-literal format strings.
  llvm::formatv(fmt, 1, 2);
}

void bool_overload() {
  llvm::formatv(false, "{0} {1}", 1, 2);
  llvm::formatv(true, "{0}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: formatv() format string requires 1 argument, but 2 arguments were provided
}
