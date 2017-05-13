//===---------------- exception_object_alignment.sh.cpp -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-exceptions

// RUN: %build -O1
// RUN: %run

// This test used to fail on Darwin because field unwindHeader of struct
// __cxa_exception and the exception object that immediately followed were not
// 16B aligned. It would segfault in class derived's constructor when a movaps
// tried to write to a memory operand that was not 16B aligned.

namespace {

struct S {
  int a;
  int __attribute__((aligned(16))) b;
};

class base1 {
protected:
  virtual ~base1() throw() {}
};

class derived: public base1 {
public:
  derived() : member() {}
private:
  S member;
};

}

int main() {
  try {
    throw derived();
  }
  catch(...) {
  }
  return 0;
}
