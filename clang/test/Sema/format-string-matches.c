// RUN: %clang_cc1 -verify -fblocks -fsyntax-only -Wformat-nonliteral -Wformat-signedness -isystem %S/Inputs %s
// RUN: %clang_cc1 -verify -fblocks -fsyntax-only -Wformat-nonliteral -Wformat-signedness -isystem %S/Inputs -fno-signed-char %s

#include <stdarg.h>

__attribute__((format(printf, 1, 2)))
int printf(const char *fmt, ...);

__attribute__((format(printf, 1, 0)))
int vprintf(const char *fmt, va_list);

// MARK: -
// Calling printf with a format from format_matches(printf) diagnoses with
// that format string
__attribute__((format_matches(printf, 1, "%s %1.5s")))
void format_str_str0(const char *fmt) {
    printf(fmt, "hello", "world");
}

__attribute__((format_matches(printf, 1, "%s" "%1.5s")))
void format_str_str1(const char *fmt) {
    printf(fmt, "hello", "world");
}

__attribute__((format_matches(printf, 1, ("%s" "%1.5s") + 5))) // expected-error{{format string is not a string literal}}
void format_str_str2(const char *fmt);

__attribute__((format_matches(printf, 1, "%s %g"))) // expected-note{{format string is defined here}}
void format_str_double_warn(const char *fmt) {
    printf(fmt, "hello", "world"); // expected-warning{{format specifies type 'double' but the argument has type 'char *'}}
}

__attribute__((format_matches(printf, 1, "%s %g")))
void vformat(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap); // XXX: ideally this would be a diagnostic
    va_end(ap);
}

// MARK: -
// Calling a function with format_matches diagnoses for incompatible formats.

void cvt_percent(const char *c) __attribute__((format_matches(printf, 1, "%%"))); // expected-note 2{{comparing with this format string}}
void cvt_at(const char *c) __attribute__((format_matches(NSString, 1, "%@")));  // \
    expected-note{{comparing with this specifier}} \
    expected-note 3{{comparing with this format string}}
void cvt_c(const char *c) __attribute__((format_matches(printf, 1, "%c"))); // expected-note{{comparing with this specifier}}
void cvt_u(const char *c) __attribute__((format_matches(printf, 1, "%u"))); // expected-note 2{{comparing with this specifier}}
void cvt_hhi(const char *c) __attribute__((format_matches(printf, 1, "%hhi")));  // expected-note 3{{comparing with this specifier}}
void cvt_i(const char *c) __attribute__((format_matches(printf, 1, "%i"))); // expected-note 4{{comparing with this specifier}}
void cvt_p(const char *c) __attribute__((format_matches(printf, 1, "%p")));
void cvt_s(const char *c) __attribute__((format_matches(printf, 1, "%s"))); // expected-note{{comparing with this specifier}}
void cvt_n(const char *c) __attribute__((format_matches(printf, 1, "%n"))); // expected-note{{comparing with this specifier}}

void test_compatibility(void) {
    cvt_c("%i");
    const char *const fmt_i = "%i";
    cvt_c(fmt_i);

    cvt_i("%c");
    cvt_c("%u"); // expected-warning{{signedness of format specifier 'u' is incompatible with 'c'}}
    cvt_u("%c"); // expected-warning{{signedness of format specifier 'c' is incompatible with 'u'}}

    const char *const fmt_c = "%c"; // expected-note{{format string is defined here}}
    cvt_u(fmt_c); // expected-warning{{signedness of format specifier 'c' is incompatible with 'u'}}

    cvt_i("%hi"); // expected-warning{{format specifier 'hi' is incompatible with 'i'}}
    cvt_i("%hhi"); // expected-warning{{format specifier 'hhi' is incompatible with 'i'}}
    cvt_i("%lli"); // expected-warning{{format specifier 'lli' is incompatible with 'i'}}
    cvt_i("%p"); // expected-warning{{format specifier 'p' is incompatible with 'i'}}
    cvt_hhi("%hhi");
    cvt_hhi("%hi"); // expected-warning{{format specifier 'hi' is incompatible with 'hhi'}} 
    cvt_hhi("%i"); // expected-warning{{format specifier 'i' is incompatible with 'hhi'}} 
    cvt_hhi("%li"); // expected-warning{{format specifier 'li' is incompatible with 'hhi'}} 
    cvt_n("%s"); // expected-warning{{format specifier 's' is incompatible with 'n'}}
    cvt_s("%hhn"); // expected-warning{{format specifier 'hhn' is incompatible with 's'}}
    
    cvt_p("%@"); // expected-warning{{invalid conversion specifier '@'}}
    cvt_at("%p"); // expected-warning{{format specifier 'p' is incompatible with '@'}}

    cvt_percent("hello");
    cvt_percent("%c"); // expected-warning{{more specifiers in format string than expected}}

    const char *const too_many = "%c"; // expected-note{{format string is defined here}}
    cvt_percent(too_many); // expected-warning{{more specifiers in format string than expected}}
}

void test_too_few_args(void) {
    cvt_at("a"); // expected-warning{{fewer specifiers in format string than expected}}
    cvt_at("%@ %@"); // expected-warning{{more specifiers in format string than expected}}

    const char *const too_few = "a"; // expected-note{{format string is defined here}}
    cvt_at(too_few); // expected-warning{{fewer specifiers in format string than expected}}
}

void cvt_several(const char *c) __attribute__((format_matches(printf, 1, "%f %i %s"))); // expected-note{{comparing with this specifier}}

void test_moving_args_around(void) {
    cvt_several("%1g %-d %1.5s"); 

    cvt_several("%3$s %1$g %2$i");

    cvt_several("%f %*s"); // expected-warning{{format argument is an indirect field width, but it should be a value}}
}

void cvt_freebsd_D(const char *c) __attribute__((format_matches(freebsd_kprintf, 1, "%D"))); // expected-note{{comparing with this specifier}}

void test_freebsd_specifiers(void) {
    cvt_freebsd_D("%D");
    cvt_freebsd_D("%b");
    cvt_freebsd_D("%s %i"); // expected-warning{{format argument is a value, but it should be an auxiliary value}}
}

// passing the wrong kind of string literal
void takes_printf_string(const char *fmt) __attribute__((format_matches(printf, 1, "%s")));
__attribute__((format_matches(freebsd_kprintf, 1, "%s"))) // expected-note{{format string is defined here}}
void takes_freebsd_kprintf_string(const char *fmt) {
    takes_printf_string(fmt); // expected-warning{{passing 'freebsd_kprintf' format string where 'printf' format string is expected}}

    const char *const fmt2 = fmt;
    takes_printf_string(fmt2); // expected-warning{{passing 'freebsd_kprintf' format string where 'printf' format string is expected}}
}

__attribute__((format_matches(printf, 1, "%s"))) // expected-note{{comparing with this specifier}}
__attribute__((format_matches(os_log, 2, "%i"))) // expected-note{{comparing with this specifier}}
void test_recv_multiple_format_strings(const char *fmt1, const char *fmt2);

__attribute__((format_matches(printf, 1, "%s")))
__attribute__((format_matches(os_log, 2, "%i")))
void test_multiple_format_strings(const char *fmt1, const char *fmt2) {
    test_recv_multiple_format_strings("%s", "%i");
    test_recv_multiple_format_strings("%s", "%s"); // expected-warning{{format specifier 's' is incompatible with 'i'}}
    test_recv_multiple_format_strings("%i", "%i"); // expected-warning{{format specifier 'i' is incompatible with 's'}}

    test_recv_multiple_format_strings(fmt1, fmt2);
    test_recv_multiple_format_strings("%.5s", fmt2);
    test_recv_multiple_format_strings(fmt1, "%04d");
    
    test_recv_multiple_format_strings("%s", fmt1); // expected-warning{{passing 'printf' format string where 'os_log' format string is expected}}
    test_recv_multiple_format_strings(fmt2, "%d"); // expected-warning{{passing 'os_log' format string where 'printf' format string is expected}}

    test_recv_multiple_format_strings(fmt2, fmt1); // \
        expected-warning{{passing 'printf' format string where 'os_log' format string is expected}} \
        expected-warning{{passing 'os_log' format string where 'printf' format string is expected}}
}

// MARK: - 
void accept_value(const char *f) __attribute__((format_matches(freebsd_kprintf, 1, "%s%i%i"))); // \
    expected-note 3{{comparing with this specifier}} \
    expected-note 3{{format string is defined here}}
void accept_indirect_field_width(const char *f) __attribute__((format_matches(freebsd_kprintf, 1, "%s%*i"))); // \
    expected-note 3{{comparing with this specifier}} \
    expected-note 3{{format string is defined here}}
void accept_indirect_field_precision(const char *f) __attribute__((format_matches(freebsd_kprintf, 1, "%s%.*i"))); // \
    expected-note 3{{comparing with this specifier}} \
    expected-note 3{{format string is defined here}}
void accept_aux_value(const char *f) __attribute__((format_matches(freebsd_kprintf, 1, "%D%i"))); // \
    expected-note 3{{comparing with this specifier}} \
    expected-note 3{{format string is defined here}}

void accept_value(const char *f) {
    accept_indirect_field_width(f); // expected-warning{{format argument is a value, but it should be an indirect field width}}
    accept_indirect_field_precision(f); // expected-warning{{format argument is a value, but it should be an indirect precision}}
    accept_aux_value(f); // expected-warning{{format argument is a value, but it should be an auxiliary value}}
}

void accept_indirect_field_width(const char *f) {
    accept_value(f); // expected-warning{{format argument is an indirect field width, but it should be a value}}
    accept_indirect_field_precision(f); // expected-warning{{format argument is an indirect field width, but it should be an indirect precision}}
    accept_aux_value(f); // expected-warning{{format argument is an indirect field width, but it should be an auxiliary value}}
}

void accept_indirect_field_precision(const char *f) {
    accept_value(f); // expected-warning{{format argument is an indirect precision, but it should be a value}}
    accept_indirect_field_width(f); // expected-warning{{format argument is an indirect precision, but it should be an indirect field width}}
    accept_aux_value(f); // expected-warning{{format argument is an indirect precision, but it should be an auxiliary value}}
}

void accept_aux_value(const char *f) {
    accept_value(f); // expected-warning{{format argument is an auxiliary value, but it should be a value}}
    accept_indirect_field_width(f); // expected-warning{{format argument is an auxiliary value, but it should be an indirect field width}}
    accept_indirect_field_precision(f); // expected-warning{{format argument is an auxiliary value, but it should be an indirect precision}}
}

// MARK: - Merging format attributes
__attribute__((format_matches(printf, 1, "%i")))
__attribute__((format_matches(printf, 1, "%d")))
void test_merge_self(const char *f);

__attribute__((format_matches(printf, 1, "%i"))) // expected-note{{comparing with this specifier}}
__attribute__((format_matches(printf, 1, "%s"))) // expected-warning{{format specifier 's' is incompatible with 'i'}}
void test_merge_self_warn(const char *f);

__attribute__((format_matches(printf, 1, "%i")))
void test_merge_redecl(const char *f);

__attribute__((format_matches(printf, 1, "%d")))
void test_merge_redecl(const char *f);

// XXX: ideally the warning and note would be swapped, but this is entirely
// reliant on which decl clang considers to be the "true one", and it might
// upset something else more important if we tried to change it.
__attribute__((format_matches(printf, 1, "%i"))) // expected-warning{{format specifier 'i' is incompatible with 's'}}
void test_merge_redecl_warn(const char *f);

__attribute__((format_matches(printf, 1, "%s"))) // expected-note{{comparing with this specifier}}
void test_merge_redecl_warn(const char *f);
