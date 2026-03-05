// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// thread_local with non-trivial destructor not implemented
// Location: CIRGenCXX.cpp:264
// Note: Simple TLS works; only destructors are NYI

#include <string>

thread_local std::string tls_string = "hello";

int test() {
  return tls_string.length();
}
