//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests for ItaniumDemangle live in llvm/unittests/Demangle

// This test exercises support for char array initializer lists added in
// dd8b266ef.
// UNSUPPORTED: using-built-library-before-llvm-20

#include "support/timer.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <string>

void test() {
  std::size_t len = 0;
  char *buf = nullptr;
  bool failed = false;
  const char* mangled = "_ZN5test71XIiEC1IdEEPT_PNS_5int_cIXplL_ZNS_4metaIiE5valueEEsrNS6_IS3_EE5valueEE4typeE";
  const char* expected =
      "test7::X<int>::X<double>(double*, test7::int_c<test7::meta<int>::value + test7::meta<double>::value>::type*)";

  int status;
  char* demang = __cxxabiv1::__cxa_demangle(mangled, buf, &len, &status);
  if (!demang || std::strcmp(demang, expected) != 0 || status != 0) {
    std::fprintf(stderr,
                 "ERROR demangling %s\n"
                 "expected: %s\n"
                 "got: %d,   %s\n",
                 mangled, expected, status, demang ? demang : "(null)");
    failed = true;
  }
  if (demang)
    buf = demang;

  free(buf);
  assert(!failed && "demangle failed");
}

void test_invalid_cases() {
  std::size_t len = 0;
  char *buf = nullptr;
  bool passed = false;
  const char* invalid =
      "Aon_PmKVPDk7?fg4XP5smMUL6;<WsI_mgbf23cCgsHbT<l8EE\0uVRkNOoXDrgdA4[8IU>Vl<>IL8ayHpiVDDDXTY;^o9;i";

  int status;
  char* demang = __cxxabiv1::__cxa_demangle(invalid, buf, &len, &status);
  if (status != -2) {
    std::printf("%s should be invalid but is not\n", invalid);
    std::printf("Got: %d, %s\n", status, demang ? demang : "(null)");
    passed = true;
  }
  if (demang)
    buf = demang;

  free(buf);
  assert(!passed && "demangle did not fail");
}

void test_invalid_args() {
  std::size_t len = 16;
  char buf[16];
  {
    // NULL mangled name should fail.
    int status;
    char* demang = __cxxabiv1::__cxa_demangle(nullptr, buf, &len, &status);
    assert(status == -3);
    assert(!demang);
  }

  {
    // Buffer without specifying length should fail.
    int status;
    char* demang = __cxxabiv1::__cxa_demangle("_Z1fv", buf, nullptr, &status);
    assert(status == -3);
    assert(!demang);
  }
}

int main(int, char**) {
  {
    timer t;
    test();
    test_invalid_cases();
    test_invalid_args();
  }
#if 0
    std::string input;
    while (std::cin)
    {
        std::getline(std::cin, input);
        if (std::cin.fail())
            break;
        std::size_t len = 0;
        int status;
        len = 0;
        char* demang = abi::__cxa_demangle(input.c_str(), 0, &len, &status);
        switch (status)
        {
        case -3:
            std::cout << "Invalid arguments\n";
            break;
        case -2:
            std::cout << "Invalid mangled name\n";
            break;
        case -1:
            std::cout << "memory allocation failure\n";
            break;
        case 0:
            std::cout << "len = " << len << '\n';
            std::cout << demang << '\n';
            std::free(demang);
            break;
        case 1:
            std::cout << "not implemented\n";
            break;
        }
        std::cout << '\n';
    }
#endif

  return 0;
}
// TODO: add cxa_demangle tests
