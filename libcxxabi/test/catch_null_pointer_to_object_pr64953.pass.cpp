//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This test case checks specifically the cases under bullet 3.3:
//
//  C++ ABI 15.3:
//  A handler is a match for an exception object of type E if
//     *  The handler is of type cv T or cv T& and E and T are the same type
//        (ignoring the top-level cv-qualifiers), or
//     *  the handler is of type cv T or cv T& and T is an unambiguous base
//        class of E, or
//  >  *  the handler is of type cv1 T* cv2 and E is a pointer type that can   <
//  >     be converted to the type of the handler by either or both of         <
//  >       o  a standard pointer conversion (4.10 [conv.ptr]) not involving   <
//  >          conversions to private or protected or ambiguous classes        <
//  >       o  a qualification conversion                                      <
//     *  the handler is a pointer or pointer to member type and E is
//        std::nullptr_t
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// This test requires the fix to https://github.com/llvm/llvm-project/issues/64953,
// which landed in d5f84e6 and is in the libc++abi built library.
// XFAIL: using-built-library-before-llvm-18

#include <exception>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

struct Base {
  int b;
};
struct Base2 {
  int b;
};
struct Derived1 : Base {
  int b;
};
struct Derived2 : Base {
  int b;
};
struct Derived3 : Base2 {
  int b;
};
struct Private : private Base {
  int b;
};
struct Protected : protected Base {
  int b;
};
struct Virtual1 : virtual Base {
  int b;
};
struct Virtual2 : virtual Base {
  int b;
};

struct Ambiguous1 : Derived1, Derived2 {
  int b;
};
struct Ambiguous2 : Derived1, Private {
  int b;
};
struct Ambiguous3 : Derived1, Protected {
  int b;
};

struct NoPublic1 : Private, Base2 {
  int b;
};
struct NoPublic2 : Protected, Base2 {
  int b;
};

struct Catchable1 : Derived3, Derived1 {
  int b;
};
struct Catchable2 : Virtual1, Virtual2 {
  int b;
};
struct Catchable3 : virtual Base, Virtual2 {
  int b;
};

// Check that, when we have a null pointer-to-object that we catch a nullptr.
template <typename T // Handler type
          ,
          typename E // Thrown exception type
          >
void assert_catches() {
  try {
    throw static_cast<E>(0);
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Statements after throw must be unreachable");
  } catch (T t) {
    assert(t == nullptr);
    return;
  } catch (...) {
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Should not have entered catch-all");
  }

  printf("%s\n", __PRETTY_FUNCTION__);
  assert(false && "The catch should have returned");
}

template <typename T // Handler type
          ,
          typename E // Thrown exception type
          >
void assert_cannot_catch() {
  try {
    throw static_cast<E>(0);
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Statements after throw must be unreachable");
  } catch (T t) {
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Should not have entered the catch");
  } catch (...) {
    assert(true);
    return;
  }

  printf("%s\n", __PRETTY_FUNCTION__);
  assert(false && "The catch-all should have returned");
}

// Check that when we have a pointer-to-actual-object we, in fact, get the
// adjusted pointer to the base class.
template <typename T // Handler type
          ,
          typename O // Object type
          >
void assert_catches_bp() {
  O* o = new (O);
  try {
    throw o;
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Statements after throw must be unreachable");
  } catch (T t) {
    assert(t == static_cast<T>(o));
    //__builtin_printf("o = %p t = %p\n", o, t);
    delete o;
    return;
  } catch (...) {
    printf("%s\n", __PRETTY_FUNCTION__);
    assert(false && "Should not have entered catch-all");
  }

  printf("%s\n", __PRETTY_FUNCTION__);
  assert(false && "The catch should have returned");
}

void f1() {
  assert_catches<Base*, Catchable1*>();
  assert_catches<Base*, Catchable2*>();
  assert_catches<Base*, Catchable3*>();
}

void f2() {
  assert_cannot_catch<Base*, Ambiguous1*>();
  assert_cannot_catch<Base*, Ambiguous2*>();
  assert_cannot_catch<Base*, Ambiguous3*>();
  assert_cannot_catch<Base*, NoPublic1*>();
  assert_cannot_catch<Base*, NoPublic2*>();
}

void f3() {
  assert_catches_bp<Base*, Catchable1>();
  assert_catches_bp<Base*, Catchable2>();
  assert_catches_bp<Base*, Catchable3>();
}

int main(int, char**) {
  f1();
  f2();
  f3();
  return 0;
}
