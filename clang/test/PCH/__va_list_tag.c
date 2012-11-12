// PR13189
// rdar://problem/11741429
// Test this without pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include %S/Inputs/__va_list_tag.h %s -emit-llvm -o -

// Test with pch.
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -emit-pch -x c-header -o %t %S/Inputs/__va_list_tag.h
// RUN: %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include-pch %t %s -verify

// expected-no-diagnostics

int myvprintf(const char *fmt, va_list args) {
    return myvfprintf(fmt, args);
}
