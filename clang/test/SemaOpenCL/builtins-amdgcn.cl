// RUN: %clang_cc1 -triple amdgcn-unknown-amdhsa -fsyntax-only -verify %s

void test_s_sleep(int x)
{
  __builtin_amdgcn_s_sleep(x); // expected-error {{argument to '__builtin_amdgcn_s_sleep' must be a constant integer}}
}
