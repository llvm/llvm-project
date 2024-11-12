// RUN: %clang_cc1 -verify -fblocks -fsyntax-only -Wformat-nonliteral -Wformat-signedness -isystem %S/Inputs %s
// RUN: %clang_cc1 -verify -fblocks -fsyntax-only -Wformat-nonliteral -Wformat-signedness -isystem %S/Inputs -fno-signed-char %s

#include <stdarg.h>

__attribute__((format(printf, 1, 2)))
int printf(const char *fmt, ...);

__attribute__((format(printf, 1, 0)))
int vprintf(const char *fmt, va_list);

__attribute__((format(scanf, 1, 2)))
int scanf(const char *fmt, ...);

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
// Calling scanf
__attribute__((format_matches(scanf, 1, "%hhi %g"))) // expected-note 3{{comparing with this specifier}}
void scan(const char *fmt) {
    char i;
    float g;
    scanf(fmt, &i, &g);
}

__attribute__((format_matches(scanf, 1, "%hhi %g"))) // expected-note{{format string is defined here}}
void scan2(const char *fmt) {
    char i;
    double g;
    scanf(fmt, &i, &g); // expected-warning{{format specifies type 'float *' but the argument has type 'double *'}}
}

void call_scan(void) {
    scan("%hhd %e");
    scan("%hd %Le"); // \
        expected-warning{{format specifier 'hd' is incompatible with 'hhi'}} \
        expected-warning{{format specifier 'Le' is incompatible with 'g'}}
    scan("%s %p"); // expected-warning{{format specifier 'p' is incompatible with 'g'}}
}

// MARK: -
// Calling a function with format_matches diagnoses for incompatible formats.

void cvt_percent(const char *c) __attribute__((format_matches(printf, 1, "%%"))); // expected-note{{comparing with this format string}}
void cvt_at(const char *c) __attribute__((format_matches(NSString, 1, "%@"))); // \
    expected-warning{{data argument not used by format string}} \
    expected-note{{comparing with this specifier}} \
    expected-note{{comparing with this format string}}
void cvt_c(const char *c) __attribute__((format_matches(printf, 1, "%c"))); // expected-note{{comparing with this specifier}}
void cvt_u(const char *c) __attribute__((format_matches(printf, 1, "%u"))); // expected-note{{comparing with this specifier}}
void cvt_i(const char *c) __attribute__((format_matches(printf, 1, "%i"))); // expected-note 2{{comparing with this specifier}}
void cvt_p(const char *c) __attribute__((format_matches(printf, 1, "%p")));
void cvt_s(const char *c) __attribute__((format_matches(printf, 1, "%s"))); // expected-note{{comparing with this specifier}}
void cvt_n(const char *c) __attribute__((format_matches(printf, 1, "%n"))); // expected-note{{comparing with this specifier}}

void test_compatibility(void) {
    cvt_c("%i");
    cvt_i("%c");
    cvt_c("%u"); // expected-warning{{signedness of format specifier 'u' is incompatible with 'c'}}
    cvt_u("%c"); // expected-warning{{signedness of format specifier 'c' is incompatible with 'u'}}

    cvt_i("%lli"); // expected-warning{{format specifier 'lli' is incompatible with 'i'}}
    cvt_i("%p"); // expected-warning{{format specifier 'p' is incompatible with 'i'}}
    cvt_n("%s"); // expected-warning{{format specifier 's' is incompatible with 'n'}}
    cvt_s("%hhn"); // expected-warning{{format specifier 'hhn' is incompatible with 's'}}
    
    cvt_p("%@"); // expected-warning{{invalid conversion specifier '@'}}
    cvt_at("%p"); // expected-warning{{format specifier 'p' is incompatible with '@'}}

    cvt_percent("hello");
    cvt_percent("%c"); // expected-warning{{fewer specifiers in format string than expected}}
}

void test_too_few_args(void) {
    cvt_at("a"); // expected-note{{comparing with this format string}}
    cvt_at("%@ %@"); // expected-warning{{fewer specifiers in format string than expected}}
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
