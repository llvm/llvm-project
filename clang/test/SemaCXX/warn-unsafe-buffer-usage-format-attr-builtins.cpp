// RUN: %clang_cc1 -Wformat -verify=expected-format %s
// RUN: %clang_cc1 -Wunsafe-buffer-usage -Wunsafe-buffer-usage-in-format-attr-call -verify=expected,expected-format %s

namespace std {
  template<typename T>
  struct basic_string {
    T* p;
    T *c_str();
    T *data();
    unsigned size_bytes();
    unsigned size();
  };

  typedef basic_string<char> string;
} // namespace std

// PR#178320 corrects the format attribute arguments for
// '__builtin_os_log_format' so that
// '-Wunsafe-buffer-usage-in-format-attr-call' behaves correctly on
// them.  For '-Wformat', the check for '__builtin_os_log_format' is
// hand-craft without using the attribute. So they are still fine.

void test_format_attr(char * Str, std::string StdStr) {
  __builtin_os_log_format(nullptr, "hello", Str); // expected-format-warning{{data argument not used by format string}}
  __builtin_os_log_format(nullptr, "hello %s", StdStr.c_str());
  __builtin_os_log_format(nullptr, "hello %s", Str);  // expected-warning{{formatting function '__builtin_os_log_format' is unsafe}} \
			         expected-note{{string argument is not guaranteed to be null-terminated}}

  __builtin_os_log_format(nullptr, "hello %"); // expected-format-warning{{incomplete format specifier}}
  __builtin_os_log_format(nullptr, "hello %d", .42); // expected-format-warning{{format specifies type 'int' but the argument has type 'double'}}
}
