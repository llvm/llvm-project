//===--------------------- catch_pointer_nullptr.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// 1b00fc5d8133 made it in the dylib in macOS 10.11
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10}}

#include <cassert>
#include <cstdio>
#include <cstdlib>

// Roll our own assertion macro to get better error messages out of the tests.
// In particular on systems that don't use __PRETTY_FUNCTION__ in assertions.
#define my_assert(pred, msg) do_assert(pred, msg, __LINE__, __PRETTY_FUNCTION__)

void do_assert(bool assert_passed, const char* msg, int line, const char* func) {
  if (assert_passed)
    return;
  std::printf("%s:%d %s: Assertion Failed '%s'\n\n", __FILE__, line, func, msg);
  std::abort();
}

struct A {};
struct Base {};
struct Derived : public Base {};

template <class To>
bool test_conversion(To) { return true; }

template <class To>
bool test_conversion(...) { return false; }

template <class Pointer>
struct CreatePointer {
  Pointer operator()() const {
      return (Pointer)0;
  }
};

template <class Tp>
struct CreatePointer<Tp*> {
  Tp* operator()() const {
      return (Tp*)42;
  }
};

template <class Throw, class Catch>
void catch_pointer_test() {
  Throw throw_ptr = CreatePointer<Throw>()();
  // Use the compiler to determine if the exception of type Throw can be
  // implicitly converted to type Catch.
  const bool can_convert = test_conversion<Catch>(throw_ptr);
  try {
    throw throw_ptr;
    assert(false);
  } catch (Catch catch_ptr) {
    Catch catch2 = CreatePointer<Catch>()();
    my_assert(can_convert, "non-convertible type incorrectly caught");
    my_assert(catch_ptr == catch2,
              "Thrown pointer does not match caught ptr");
  } catch (...) {
    my_assert(!can_convert, "convertible type incorrectly not caught");
  }
}

// Generate CV qualified pointer typedefs.
template <class Tp, bool First = false>
struct TestTypes {
  typedef Tp* Type;
  typedef Tp const* CType;
  typedef Tp volatile* VType;
  typedef Tp const volatile* CVType;
};

// Special case for cv-qualifying a pointer-to-member without adding an extra
// pointer to it.
template <class Member, class Class>
struct TestTypes<Member Class::*, true> {
  typedef Member (Class::*Type);
  typedef const Member (Class::*CType);
  typedef volatile Member (Class::*VType);
  typedef const volatile Member (Class::*CVType);
};

template <class Throw, class Catch, int level, bool first = false>
struct generate_tests_imp {
  typedef TestTypes<Throw, first> ThrowTypes;
  typedef TestTypes<Catch, first> CatchTypes;
  void operator()() {
      typedef typename ThrowTypes::Type Type;
      typedef typename ThrowTypes::CType CType;
      typedef typename ThrowTypes::VType VType;
      typedef typename ThrowTypes::CVType CVType;

      run_catch_tests<Type>();
      run_catch_tests<CType>();
      run_catch_tests<VType>();
      run_catch_tests<CVType>();
  }

  template <class ThrowTp>
  void run_catch_tests() {
      typedef typename CatchTypes::Type Type;
      typedef typename CatchTypes::CType CType;
      typedef typename CatchTypes::VType VType;
      typedef typename CatchTypes::CVType CVType;

      catch_pointer_test<ThrowTp, Type>();
      catch_pointer_test<ThrowTp, CType>();
      catch_pointer_test<ThrowTp, VType>();
      catch_pointer_test<ThrowTp, CVType>();

      generate_tests_imp<ThrowTp, Type, level-1>()();
      generate_tests_imp<ThrowTp, CType, level-1>()();
      generate_tests_imp<ThrowTp, VType, level-1>()();
      generate_tests_imp<ThrowTp, CVType, level-1>()();
  }
};

template <class Throw, class Catch, bool first>
struct generate_tests_imp<Throw, Catch, 0, first> {
  void operator()() {
      catch_pointer_test<Throw, Catch>();
  }
};

template <class Throw, class Catch, int level>
struct generate_tests : generate_tests_imp<Throw, Catch, level, true> {};

int main(int, char**)
{
  generate_tests<int, int, 3>()();
  generate_tests<Base, Derived, 2>()();
  generate_tests<Derived, Base, 2>()();
  generate_tests<int, void, 2>()();
  generate_tests<void, int, 2>()();

  generate_tests<int A::*, int A::*, 3>()();
  generate_tests<int A::*, void, 2>()();
  generate_tests<void, int A::*, 2>()();

  return 0;
}
