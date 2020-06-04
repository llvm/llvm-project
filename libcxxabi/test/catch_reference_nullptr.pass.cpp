//===--------------------- catch_pointer_nullptr.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03,
// UNSUPPORTED: no-exceptions

#include <cassert>
#include <cstdlib>

struct A {};

template<typename T, bool CanCatchNullptr>
static void catch_nullptr_test() {
  try {
    throw nullptr;
  } catch (T &p) {
    assert(CanCatchNullptr && !static_cast<bool>(p));
  } catch (...) {
    assert(!CanCatchNullptr);
  }
}

int main()
{
  using nullptr_t = decltype(nullptr);

  // A reference to nullptr_t can catch nullptr.
  catch_nullptr_test<nullptr_t, true>();
  catch_nullptr_test<const nullptr_t, true>();
  catch_nullptr_test<volatile nullptr_t, true>();
  catch_nullptr_test<const volatile nullptr_t, true>();

  // No other reference type can.
#if 0
  // FIXME: These tests fail, because the ABI provides no way for us to
  // distinguish this from catching by value.
  catch_nullptr_test<void *, false>();
  catch_nullptr_test<void * const, false>();
  catch_nullptr_test<int *, false>();
  catch_nullptr_test<A *, false>();
  catch_nullptr_test<int A::*, false>();
  catch_nullptr_test<int (A::*)(), false>();
#endif
}
