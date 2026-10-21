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
  llvm::formatv("{0:$[,]}", 1);
  llvm::formatv("{ 0 }", 1);
  llvm::formatv("{  0  :x}", 1);
  llvm::formatv("{ 0 } { 1 }", 1, 2);
  llvm::formatv("no replacements");
  llvm::formatv("escaped {{ braces }}");
  llvm::formatv("{}", 1);
  llvm::formatv("{} {}", 1, 2);
  llvm::formatv(false, "{0}", 1);
}

void too_few_args() {
  llvm::formatv("{0}");
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: format string requires 1 argument, but 0 arguments were provided

  llvm::formatv("{0} {1}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: format string requires 2 arguments, but 1 argument was provided

  llvm::formatv("{0} {1} {2}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: format string requires 3 arguments, but 2 arguments were provided
}

void too_many_args() {
  llvm::formatv("{0}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: argument unused in format string

  llvm::formatv("no replacements", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: argument unused in format string
}

void mixed_indices() {
  llvm::formatv("{} {1}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: format string mixes automatic and explicit indices
}

void holes_in_indices() {
  llvm::formatv("{0} {2}", 1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: argument unused in format string

  llvm::formatv("{2}", 1, 2, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: argument unused in format string
  // CHECK-MESSAGES: :[[@LINE-2]]:27: warning: argument unused in format string

  llvm::formatv("{1}", 10, 20, 30);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: argument unused in format string
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: argument unused in format string
}

void non_literal_format_string(const char *fmt) {
  // No warning for non-literal format strings.
  llvm::formatv(fmt, 1, 2);
}

void bool_overload() {
  llvm::formatv(false, "{0} {1}", 1, 2);
  llvm::formatv(true, "{0}", 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: argument unused in format string
}

void invalid_index() {
  llvm::formatv("{abc}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid replacement index in format string
}
