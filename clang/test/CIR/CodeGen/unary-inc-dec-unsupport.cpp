// RUN: %clang_cc1 -triple powerpc64le -fclangir -emit-cir -verify %s -o -

// PPC fp128 (__ibm128) unary inc/dec is not yet supported, drop the errorNYI
// and turn this into a regular test.

__ibm128 g; // expected-error@*:* {{ClangIR code gen Not Yet Implemented: processing of built-in type: '__ibm128'}}

void test_pre_inc() { ++g; }  
void test_pre_dec() { --g; }
void test_post_inc() { g++; }
void test_post_dec() { g--; }
