// RUN: %check_clang_tidy %s llvm-formatv-string %t -- \
// RUN:   -config='{CheckOptions: {llvm-formatv-string.AdditionalFunctions: "llvm::createStringErrorV"}}'

namespace llvm {

template <typename... Ts>
void createStringErrorV(int EC, const char *Fmt, Ts &&...Vals) {}

template <typename... Ts>
void createStringErrorV(const char *Fmt, Ts &&...Vals) {}

} // namespace llvm

void correct() {
  llvm::createStringErrorV(0, "{0} {1}", 1, 2);
  llvm::createStringErrorV("{0}", 42);
}

void wrong_count() {
  llvm::createStringErrorV(0, "{0} {1}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: formatv() format string requires 2 arguments, but 1 argument was provided

  llvm::createStringErrorV("{0} {1}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: formatv() format string requires 2 arguments, but 1 argument was provided
}
