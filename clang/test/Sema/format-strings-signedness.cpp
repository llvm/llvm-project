// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wformat -Wformat-signedness %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -verify -Wformat -Wformat-signedness %s

// Verify that -Wformat-signedness alone (without -Wformat) trigger the
// warnings. Note in gcc this will not trigger the signedness warnings as
// -Wformat is default off in gcc.
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -verify -Wformat-signedness %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -verify -Wformat-signedness %s

// Verify that -Wformat-signedness warnings are not reported with only -Wformat
// (gcc compat).
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wformat -verify=okay %s

// Verify that -Wformat-signedness with -Wno-format are not reported (gcc compat).
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wformat-signedness -Wno-format -verify=okay %s
// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -fsyntax-only -Wno-format -Wformat-signedness -verify=okay %s
// okay-no-diagnostics

// Ignore 'GCC requires a function with the 'format' attribute to be variadic'.
#pragma GCC diagnostic ignored "-Wgcc-compat"
namespace GH161075 {
template <typename... Args>
void format(const char *fmt, Args &&...args)
    __attribute__((format(printf, 1, 2)));

void do_format() {
  bool b = false;
  format("%hhi %hhu %hi %hu %i %u", b, b, b, b, b, b);  // expected-warning {{format specifies type 'unsigned char' but the argument has type 'bool', which differs in signedness}}
}
} // namespace GH161075
