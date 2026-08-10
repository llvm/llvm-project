// RUN: %clang_analyze_cc1 -analyzer-checker=core,security.cert.env.InvalidPtr -verify %s

// expected-no-diagnostics

namespace other {
int strerror(int errnum); // custom strerror
void no_crash_on_custom_strerror() {
  (void)strerror(0); // no-crash
}
} // namespace other
