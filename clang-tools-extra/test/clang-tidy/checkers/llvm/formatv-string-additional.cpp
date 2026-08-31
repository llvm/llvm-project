// RUN: %check_clang_tidy %s llvm-formatv-string %t -- \
// RUN:   -config='{CheckOptions: {llvm-formatv-string.AdditionalFunctions: "mylib::log"}}'

namespace llvm {

template <typename... Ts>
void formatv(const char *Fmt, Ts &&...Vals) {}

} // namespace llvm

namespace mylib {

enum Level { Info, Error };

template <typename... Ts>
void log(Level L, const char *Fmt, Ts &&...Vals) {}

} // namespace mylib

void correct() {
  mylib::log(mylib::Info, "{0} {1}", 1, 2);
  mylib::log(mylib::Error, "{0}", 42);
}

void wrong_count() {
  mylib::log(mylib::Info, "{0} {1}", 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: format string requires 2 arguments, but 1 argument was provided
}
