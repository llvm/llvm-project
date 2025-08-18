
// RUN: %clang_cc1 -x c -Wstring-concatenation -fsyntax-only -verify %s
// RUN: %clang_cc1 -x c++ -Wstring-concatenation -fsyntax-only -verify %s

const char *missing_comma[] = {
    "basic_filebuf",
    "basic_ios",
    "future",
    "optional",
    "packaged_task" // expected-note{{place parentheses around the string literal to silence warning}}
    "promise",      // expected-warning{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}     
    "shared_future"
};

#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif

const wchar_t *missing_comma_wchar[] = {
    L"basic_filebuf",
    L"packaged_task" // expected-note{{place parentheses around the string literal to silence warning}}
    L"promise",      // expected-warning{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
    L"shared_future"
};

#if __cplusplus >= 201103L
const char *missing_comma_u8[] = {
    u8"basic_filebuf",
    u8"packaged_task" // expected-note{{place parentheses around the string literal to silence warning}}
    u8"promise",      // expected-warning{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
    u8"shared_future"
};
#endif

const char *missing_comma_same_line[] = {"basic_filebuf", "basic_ios",
                       "future" "optional",         // expected-note{{place parentheses around the string literal to silence warning}}
                       "packaged_task", "promise"}; // expected-warning@-1{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}

const char *missing_comma_different_lines[] = {"basic_filebuf", "basic_ios" // expected-note{{place parentheses around the string literal to silence warning}}
                       "future", "optional",        // expected-warning{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
                       "packaged_task", "promise"};

const char *missing_comma_same_line_all_literals[] = {"basic_filebuf", "future" "optional", "packaged_task"}; // expected-note{{place parentheses around the string literal to silence warning}}
                                                                               // expected-warning@-1{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}

char missing_comma_inner[][5] = {
    "a",
    "b",
    "c" // expected-note{{place parentheses around the string literal to silence warning}}
    "d" // expected-warning{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
};

const char *warn[] = { "cpll", "gpll", "hdmiphy" "usb480m" }; // expected-note{{place parentheses around the string literal to silence warning}}
// expected-warning@-1{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}

const char *missing_two_commas_ignore[] = {"basic_filebuf",
                       "basic_ios" 
                       "future"  
                       "optional",
                       "packaged_task"};

#define ONE(x) x
#define TWO "foo"
const char *macro_test[] = { ONE("foo"),
                             TWO,
                             "foo" TWO // expected-note{{place parentheses around the string literal to silence warning}}
                           };          // expected-warning@-1{{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}

// Do not warn for macros.

#define BASIC_IOS "basic_ios"
#define FUTURE "future"
const char *macro_test2[] = {"basic_filebuf", BASIC_IOS
                        FUTURE, "optional",
                       "packaged_task", "promise"};

#define FOO(xx) xx "_normal", \
                xx "_movable",

const char *macro_test3[] = {"basic_filebuf",
                       "basic_ios",
                       FOO("future")
                       "optional",
                       "packaged_task"};

#define BAR(name) #name "_normal"

const char *macro_test4[] = {"basic_filebuf",
                       "basic_ios",
                       BAR(future),
                       "optional",
                       "packaged_task"};

#define SUPPRESS(x) x
const char *macro_test5[] = { SUPPRESS("foo" "bar"), "baz" };

typedef struct {
    int i;
    const char s[11];
} S;

S s = {1, "hello" "world"};

const char *not_warn[] = {
    "hello"
    "world", "test"
};

const char *not_warn2[] = {
    "// Aaa\\\n"   " Bbb\\ \n"   " Ccc?" "?/\n",
    "// Aaa\\\r\n" " Bbb\\ \r\n" " Ccc?" "?/\r\n",
    "// Aaa\\\r"   " Bbb\\ \r"   " Ccc?" "?/\r"
};

const char *not_warn3[] = {
  "// \\tparam aaa Bbb\n",
  "// \\tparam\n"
  "//     aaa Bbb\n",
  "// \\tparam \n"
  "//     aaa Bbb\n",
  "// \\tparam aaa\n"
  "// Bbb\n"
};

const char *not_warn4[] =  {"title",
               "aaaa "
               "bbbb "
               "cccc "
               "ddd.",
               "url"
};

typedef struct {
  const char *a;
  const char *b;
  const char *c;
} A;

const A not_warn5 = (A){"",
                        ""
                        "",
                        ""};

#ifdef __cplusplus
const A not_warn6 =  A{"",
                      ""
                      "",
                      ""};
#endif

static A not_warn7 = {"",

  ""
  "",
  ""};


// Do not warn when all the elements in the initializer are concatenated together.
const char *all_elems_in_init_concatenated[] = {"a" "b" "c"};

// Warning can be supressed also by extra parentheses.
const char *extra_parens_to_suppress_warning[] = {
    "basic_filebuf",
    "basic_ios",
    "future",
    "optional",
    ("packaged_task"
    "promise"),
    "shared_future"
};

const char *multiple_missing_commas1[] = {
    "1",
    "2"  // expected-note {{place parentheses around the string literal to silence warning}}
    "3", // expected-warning {{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
    "4",
    "5",
    "6"  // expected-note {{place parentheses around the string literal to silence warning}}
    "7", // expected-warning {{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
    "8",
    "9",
    "10",
    "11",
};

const char *multiple_missing_commas2[] = {
    "1",
    "2"
    "3"
    "4"
    "5",
    "6"  // expected-note {{place parentheses around the string literal to silence warning}}
    "7", // expected-warning {{suspicious concatenation of string literals in an array initialization; did you mean to separate the elements with a comma?}}
    "8",
    "9",
    "10",
    "11",
};
