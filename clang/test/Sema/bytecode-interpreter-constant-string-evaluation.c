// RUN: %clang_cc1 -x c   -std=c2x   -fsyntax-only -verify -Wno-string-plus-int -Wno-unused-value %s
// RUN: %clang_cc1 -x c   -std=c2x   -fsyntax-only -verify -Wno-string-plus-int -Wno-unused-value %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify -Wno-string-plus-int -Wno-unused-value %s
// RUN: %clang_cc1 -x c++ -std=c++23 -fsyntax-only -verify -Wno-string-plus-int -Wno-unused-value %s -fexperimental-new-constant-interpreter

void test(char* c) {
  __builtin_strcat(c, "42" + 0);
  __builtin_strcat(c, "42" + 1);
  __builtin_strcat(c, "42" + 2);
  __builtin_strcat(c, "42" + 3);
  __builtin_strcat(c, "42" + 4);
  char buffer[10];
  __builtin_sprintf(buffer, "%%d%d%d"+0, 1);
  // expected-warning@-1 {{more '%' conversions than data arguments}}
  __builtin_sprintf(buffer, "%%d%d%d"+1, 1);
  // expected-warning@-1 {{more '%' conversions than data arguments}}
  __builtin_sprintf(buffer, "%%d%d%d"+2, 1);
  // expected-warning@-1 {{more '%' conversions than data arguments}}
  __builtin_sprintf(buffer, "%%d%d%d"+3, 1);
  // expected-warning@-1 {{more '%' conversions than data arguments}}
  __builtin_sprintf(buffer, "%%d%d%d"+4, 1);
  __builtin_sprintf(buffer, "%%d%d%d"+5, 1);
  __builtin_sprintf(buffer, "%%d%d%d"+6, 1);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+7, 1);
  // expected-warning@-1 {{format string is empty}}
  __builtin_sprintf(buffer, "%%d%d%d"+8, 1);
  // TODO: we should probably warning about the format string being out of bounds

  __builtin_sprintf(buffer, "%%d%d%d"+0, 1, 2);
  __builtin_sprintf(buffer, "%%d%d%d"+1, 1, 2);
  // expected-warning@-1 {{more '%' conversions than data arguments}}
  __builtin_sprintf(buffer, "%%d%d%d"+2, 1, 2);
  __builtin_sprintf(buffer, "%%d%d%d"+3, 1, 2);
  __builtin_sprintf(buffer, "%%d%d%d"+4, 1, 2);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+5, 1, 2);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+6, 1, 2);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+7, 1, 2);
  // expected-warning@-1 {{format string is empty}}
  __builtin_sprintf(buffer, "%%d%d%d"+8, 1, 2);
  __builtin_sprintf(buffer, "%%d%d%d"+9, 1, 2);

  __builtin_sprintf(buffer, "%%d%d%d"+0, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+1, 1, 2, 3);
  __builtin_sprintf(buffer, "%%d%d%d"+2, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+3, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+4, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+5, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+6, 1, 2, 3);
  // expected-warning@-1 {{data argument not used by format string}}
  __builtin_sprintf(buffer, "%%d%d%d"+7, 1, 2, 3);
  // expected-warning@-1 {{format string is empty}}
  __builtin_sprintf(buffer, "%%d%d%d"+8, 1, 2, 3);
  __builtin_sprintf(buffer, "%%d%d%d"+9, 1, 2, 3);
  static const char format_string[] = {'%', '%', 'd', '%', 'd', '%', 'd'};
  __builtin_sprintf(buffer, format_string+0, 1);
  __builtin_sprintf(buffer, format_string+1, 1);
  __builtin_sprintf(buffer, format_string+2, 1);
  __builtin_sprintf(buffer, format_string+3, 1);
  __builtin_sprintf(buffer, format_string+4, 1);
  __builtin_sprintf(buffer, format_string+5, 1);
  __builtin_sprintf(buffer, format_string+6, 1);
  __builtin_sprintf(buffer, format_string+7, 1);
  #ifdef __cplusplus
  static constexpr char ce_format_string[] = {'%', '%', 'd', '%', 'd', '%', 'd'};
  __builtin_sprintf(buffer, ce_format_string+0, 1);
  __builtin_sprintf(buffer, ce_format_string+1, 1);
  __builtin_sprintf(buffer, ce_format_string+2, 1);
  __builtin_sprintf(buffer, ce_format_string+3, 1);
  __builtin_sprintf(buffer, ce_format_string+4, 1);
  __builtin_sprintf(buffer, ce_format_string+5, 1);
  __builtin_sprintf(buffer, ce_format_string+6, 1);
  __builtin_sprintf(buffer, ce_format_string+7, 1);
  #endif
}

#ifdef __cplusplus
template <class FormatStringSource> bool test_template() {
  char buffer[10];
  __builtin_sprintf(buffer, FormatStringSource::format(0), 1); // #template_test1
  __builtin_sprintf(buffer, FormatStringSource::format(1), 1); // #template_test2
  __builtin_sprintf(buffer, FormatStringSource::format(2), 1); // #template_test3
  __builtin_sprintf(buffer, FormatStringSource::format(3), 1); // #template_test4
  __builtin_sprintf(buffer, FormatStringSource::format(4), 1); // #template_test5
  __builtin_sprintf(buffer, FormatStringSource::format(5), 1); // #template_test6
  __builtin_sprintf(buffer, FormatStringSource::format(6), 1); // #template_test7
  __builtin_sprintf(buffer, FormatStringSource::format(7), 1); // #template_test8
  __builtin_sprintf(buffer, FormatStringSource::format(8), 1); // #template_test9
  return true;
}

struct LiteralFormatStr {
  static consteval const char *format(int N) {
    return "%%d%d%d" + N; // #LiteralFormatStrLiteral
  }
};

struct ConstLiteralFormatStr {
  static constexpr const char *formatStr = "%%d%d%d"; // #ConstLiteralFormatStrLiteral
  static consteval const char *format(int N) {
    return formatStr + N;
  }
};

struct NullTerminatedArrayFormatStr {
  static constexpr char formatStr[] = {'%', '%', 'd', '%', 'd', '%', 'd', 0};
  static consteval const char *format(int N) {
    return formatStr + N;
  }
};

struct NoNullTerminatedArrayFormatStr {
  static constexpr char formatStr[] = {'%', '%', 'd', '%', 'd', '%', 'd'};
  static consteval const char *format(int N) {
    return formatStr + N; // #NoNullTerminatedArrayFormatStr_format
  }
};

void test_templates() {
  test_template<LiteralFormatStr>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_template<LiteralFormatStr>' requested here}}
  // expected-warning@#template_test1 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test2 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test3 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test4 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test5 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test6 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test7 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test8 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test9 {{more '%' conversions than data arguments}}
  // expected-note@#LiteralFormatStrLiteral 9 {{format string is defined here}}
  test_template<ConstLiteralFormatStr>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_template<ConstLiteralFormatStr>' requested here}}
  // expected-warning@#template_test1 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test2 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test3 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test4 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test5 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test6 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test7 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test8 {{more '%' conversions than data arguments}}
  // expected-warning@#template_test9 {{more '%' conversions than data arguments}}
  // expected-note@#ConstLiteralFormatStrLiteral 9 {{format string is defined here}}
  test_template<NullTerminatedArrayFormatStr>();
  test_template<NoNullTerminatedArrayFormatStr>();
  // expected-note@-1 {{in instantiation of function template specialization 'test_template<NoNullTerminatedArrayFormatStr>' requested here}}
  // expected-note@#NoNullTerminatedArrayFormatStr_format {{cannot refer to element 8 of array of 7 elements in a constant expression}}
  // expected-error@#template_test9 {{call to consteval function 'NoNullTerminatedArrayFormatStr::format' is not a constant expression}}
  // expected-note@#template_test9 {{in call to 'format(8)'}}
}

#endif
