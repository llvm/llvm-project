/* Tests for interactions between C++ and assert.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Undefine NDEBUG to ensure the build system e.g. CFLAGS/CXXFLAGS
   does not disable the asserts we want to test.  */
#undef NDEBUG
#include <assert.h>

/* The C++ standard requires that if the assert argument is a constant
   subexpression, then the assert itself is one, too.  */
constexpr int
check_constexpr ()
{
  return (assert (true), 1);
}

/* Objects of this class can be contextually converted to bool, but
   cannot be compared to int.  */
struct no_int
{
  no_int () = default;
  no_int (const no_int &) = delete;

  explicit operator bool () const
  {
    return true;
  }

  bool operator! () const; /* No definition.  */
  template <class T> bool operator== (T) const; /* No definition.  */
  template <class T> bool operator!= (T) const; /* No definition.  */
};

/* This class tests that operator== is not used by assert.  */
struct bool_and_int
{
  bool_and_int () = default;
  bool_and_int (const no_int &) = delete;

  explicit operator bool () const
  {
    return true;
  }

  bool operator! () const; /* No definition.  */
  template <class T> bool operator== (T) const; /* No definition.  */
  template <class T> bool operator!= (T) const; /* No definition.  */
};

static int
do_test ()
{
  {
    no_int value;
    assert (value);
  }

  {
    bool_and_int value;
    assert (value);
  }

  return 0;
}

#include <support/test-driver.c>
