//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_put<charT, OutputIterator>

// iter_type put(iter_type s, ios_base& iob, char_type fill, long double v) const;

// XFAIL: win32-broken-printf-a-precision

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include <cmath>
#include "test_macros.h"
#include "test_iterators.h"

#ifdef _AIX
#  define LC_SUFFIX ".0"
#  define LG_SUFFIX ";0"
#  define SP_LC_SUFFIX ".0"
#  define SP_LG_SUFFIX ";0"
#  define PADDING "****************"
#  define SP_PADDING "****************"
#else
#  define LC_SUFFIX ""
#  define LG_SUFFIX ""
#  define SP_LC_SUFFIX "."
#  define SP_LG_SUFFIX ";"
#  define PADDING "******************"
#  define SP_PADDING "*****************"
#endif

typedef std::num_put<char, cpp17_output_iterator<char*> > F;

class my_facet : public F {
public:
  explicit my_facet(std::size_t refs = 0) : F(refs) {}
};

class my_numpunct : public std::numpunct<char> {
public:
  my_numpunct() : std::numpunct<char>() {}

protected:
  virtual char_type do_decimal_point() const { return ';'; }
  virtual char_type do_thousands_sep() const { return '_'; }
  virtual std::string do_grouping() const { return std::string("\1\2\3"); }
};

void test1() {
  char str[200];
  std::locale lc = std::locale::classic();
  std::locale lg(lc, new my_numpunct);
  const my_facet f(1);
  {
    long double v = -0.;
    std::ios ios(0);
    std::hexfloat(ios);
    // %a
    {
      ios.precision(0);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
      ios.precision(1);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
      ios.precision(6);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LC_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" LG_SUFFIX "p+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0x0" LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LC_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LC_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0x0" SP_LG_SUFFIX "p+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0x0" SP_LG_SUFFIX "p+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LC_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" LG_SUFFIX "P+0" PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == PADDING "-0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" PADDING "0X0" LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LC_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LC_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-0X0" SP_LG_SUFFIX "P+0" SP_PADDING);
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == SP_PADDING "-0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "-" SP_PADDING "0X0" SP_LG_SUFFIX "P+0");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
      ios.precision(16);
      {
      }
      ios.precision(60);
      {
      }
    }
  }
}

void test2() {
  std::locale lc = std::locale::classic();
  std::locale lg(lc, new my_numpunct);
#if (defined(__APPLE__) || defined(TEST_HAS_GLIBC) || defined(__MINGW32__)) && defined(TEST_LONG_DOUBLE_IS_80_BIT)
  // This test assumes that long doubles are x87 80 bit long doubles, and
  // assumes one specific way of formatting the long doubles. (There are
  // multiple valid ways of hex formatting the same float.)
  //
  // FreeBSD does use x87 80 bit long doubles, but normalizes the hex floats
  // differently.
  //
  // This test assumes the form used by Glibc, Darwin and others, where the
  // 64 mantissa bits are grouped by nibble as they are stored in the long
  // double representation (nibble aligned at the end of the least significant
  // bits). This makes 1.0L to be formatted as "0x8p-3" (where the leading
  // bit of the mantissa is the highest bit in the 0x8 nibble), and makes
  // __LDBL_MAX__ be formatted as "0xf.fffffffffffffffp+16380".
  //
  // FreeBSD normalizes/aligns the leading bit of the mantissa as a separate
  // nibble, so that 1.0L is formatted as "0x1p+0" and __LDBL_MAX__ as
  // "0x1.fffffffffffffffep+16383" (note the lowest bit of the last nibble
  // being zero as the nibbles don't align to the actual floats).
  const my_facet f(1);
  char str[200];
  {
    long double v = 1234567890.125;
    std::ios ios(0);
    std::hexfloat(ios);
    // %a
    {
      ios.precision(0);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.width(0);
              ios.imbue(lc);
              {
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
      ios.precision(1);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
      ios.precision(6);
      {
      }
      ios.precision(16);
      {
      }
      ios.precision(60);
      {
        std::nouppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9.32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x9;32c05a44p+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0x*********9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9.32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9.32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0x9;32c05a44p+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0x9;32c05a44p+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
        std::uppercase(ios);
        {
          std::noshowpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9.32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X9;32C05A44P+27*********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "*********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "0X*********9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
          std::showpos(ios);
          {
            std::noshowpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
            std::showpoint(ios);
            {
              ios.imbue(lc);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9.32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9.32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
              ios.imbue(lg);
              {
                ios.width(0);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::left(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+0X9;32C05A44P+27********");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::right(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "********+0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
                ios.width(25);
                std::internal(ios);
                {
                  cpp17_output_iterator<char*> iter = f.put(cpp17_output_iterator<char*>(str), ios, '*', v);
                  std::string ex(str, base(iter));
                  assert(ex == "+********0X9;32C05A44P+27");
                  assert(ios.width() == 0);
                }
              }
            }
          }
        }
      }
    }
  }
#endif
}

int main(int, char**) {
  test1();
  test2();

  return 0;
}
