//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Catching an exception thrown as nullptr was not properly handled before
// 2f984cab4fa7, which landed in macOS 10.13
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// UNSUPPORTED: c++03
// UNSUPPORTED: no-exceptions

#include <cassert>
#include <cstdlib>

struct A {};

void test1()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (int* p)
    {
        assert(!p);
    }
    catch (long*)
    {
        assert(false);
    }
}

void test2()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A* p)
    {
        assert(!p);
    }
    catch (int*)
    {
        assert(false);
    }
}

template <class Catch>
void catch_nullptr_test() {
  try {
    throw nullptr;
    assert(false);
  } catch (Catch c) {
    assert(!c);
  } catch (...) {
    assert(false);
  }
}


int main(int, char**)
{
  // catch naked nullptrs
  test1();
  test2();

  catch_nullptr_test<int*>();
  catch_nullptr_test<int**>();
  catch_nullptr_test<int A::*>();
  catch_nullptr_test<const int A::*>();
  catch_nullptr_test<int A::**>();

  return 0;
}
