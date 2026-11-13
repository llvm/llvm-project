//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The fix for LWG2381 (https://github.com/llvm/llvm-project/pull/77948) changed behavior of
// FP parsing. This requires 3e15c97fa3812993bdc319827a5c6d867b765ae8 in the dylib.
// Subnormal stream extraction also requires the PR38360 fix in the dylib.
// XFAIL: using-built-library-before-llvm-24
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <locale>

// class num_get<charT, InputIterator>

// iter_type get(iter_type in, iter_type end, ios_base&,
//               ios_base::iostate& err, long double& v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include "test_macros.h"
#include "test_iterators.h"
#include "hexfloat.h"

typedef std::num_get<char, cpp17_input_iterator<const char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

void test_value(const std::string& str,
                long double expected,
                std::ios_base::iostate expected_err = std::ios_base::goodbit) {
  const my_facet f(1);
  std::ios ios(0);
  long double v                          = -1;
  std::ios_base::iostate err             = std::ios_base::goodbit;
  const char* first                      = str.c_str();
  cpp17_input_iterator<const char*> iter = f.get(
      cpp17_input_iterator<const char*>(first), cpp17_input_iterator<const char*>(first + str.size() + 1), ios, err, v);
  assert(base(iter) == first + str.size());
  assert(err == expected_err);
  assert(v == expected);
  assert(std::signbit(v) == std::signbit(expected));

  std::istringstream is(str);
  v = -1;
  is >> v;
  assert(is.rdstate() == (expected_err | std::ios_base::eofbit));
  assert(v == expected);
  assert(std::signbit(v) == std::signbit(expected));
}

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    long double v = -1;
    {
        const char str[] = "123";
        assert((ios.flags() & ios.basefield) == ios.dec);
        assert(ios.getloc().name() == "C");
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 123);
    }
    {
        const char str[] = "-123";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == -123);
    }
    {
        const char str[] = "123.5";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 123.5);
    }
    {
        const char str[] = "125e-1";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 125e-1);
    }
    {
        const char str[] = "0x125p-1";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == hexfloat<long double>(0x125, 0, -1));
    }
    {
        const char str[] = "inf";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
        const char str[] = "INF";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
        const char str[] = "-inf";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str + 1);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
        const char str[] = "-INF";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str + 1);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
        const char str[] = "nan";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
        const char str[] = "NAN";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str);
        assert(err == ios.failbit);
        assert(v == 0.0l);
    }
    {
      const char str[] = "p00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "P00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "+p00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "+P00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "-p00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "-P00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "e00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "E00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "+e00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "+E00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "-e00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
      const char str[] = "-E00";
      std::hex(ios);
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0.0l);
    }
    {
        const char str[] = "1.189731495357231765021264e+49321";
        std::ios_base::iostate err = ios.goodbit;
        v = 0;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "1.189731495357231765021264e+49329";
        std::ios_base::iostate err = ios.goodbit;
        v = 0;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "11.189731495357231765021264e+4932";
        std::ios_base::iostate err = ios.goodbit;
        v = 0;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "91.189731495357231765021264e+4932";
        std::ios_base::iostate err = ios.goodbit;
        v = 0;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "304888344611713860501504000000";
        std::ios_base::iostate err = ios.goodbit;
        v = 0;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err != ios.failbit);
        assert(v == 304888344611713860501504000000.0L);
    }
    {
        v = -1;
        const char str[] = "1.19973e+4933"; // unrepresentable
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == HUGE_VALL);
    }
    {
        v = -1;
        const char str[] = "-1.18974e+4932"; // unrepresentable
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == -HUGE_VALL);
    }
    {
        v = -1;
        const char str[] = "2-";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+1);
        assert(err == ios.goodbit);
        assert(v == 2);
    }
    {
      v                                      = -1;
      const char str[]                       = ".5";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 2);
      assert(err == ios.goodbit);
      assert(v == 0.5l);
    }
    {
      v                                      = -1;
      const char str[]                       = "-.5";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 3);
      assert(err == ios.goodbit);
      assert(v == -0.5l);
    }
    {
      v                                      = -1;
      const char str[]                       = ".5E1";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 4);
      assert(err == ios.goodbit);
      assert(v == 5.0l);
    }
    {
      v                                      = -1;
      const char str[]                       = "-.5e+1";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 6);
      assert(err == ios.goodbit);
      assert(v == -5.0l);
    }
    {
      v                                      = -1;
      const char str[]                       = ".625E-1";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 7);
      assert(err == ios.goodbit);
      assert(v == 0.0625l);
    }
    {
      v                                      = -1;
      const char str[]                       = "-.3125e-1";
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + sizeof(str)), ios, err, v);
      assert(base(iter) == str + 9);
      assert(err == ios.goodbit);
      assert(v == -0.03125l);
    }

    // Underflow to zero still reports failure.
    test_value("1e-99999", 0, std::ios_base::failbit);
    test_value("-1e-99999", -0.0, std::ios_base::failbit);

    // Nonzero subnormals are representable (https://llvm.org/PR38360).
    {
      typedef std::numeric_limits<long double> limits;
      const long double values[] = {limits::denorm_min(), -limits::denorm_min(), limits::min() / 2, -limits::min() / 2};
      for (unsigned i = 0; i != sizeof(values) / sizeof(values[0]); ++i) {
        std::ostringstream os;
        os.precision(limits::digits10 + 3);
        os << values[i];
        test_value(os.str(), values[i]);
      }
    }

  return 0;
}
