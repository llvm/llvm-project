// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

// NOTE: This file intentionally uses DOS-style line endings to test
// that we don't propagate them into string literals as per [lex.string]p4.

constexpr const char* p = R"(a\
b
c)";

static_assert(p[0] == 'a',  "");
static_assert(p[1] == '\\', "");
static_assert(p[2] == '\n', "");
static_assert(p[3] == 'b',  "");
static_assert(p[4] == '\n', "");
static_assert(p[5] == 'c',  "");
static_assert(p[6] == '\0', "");
