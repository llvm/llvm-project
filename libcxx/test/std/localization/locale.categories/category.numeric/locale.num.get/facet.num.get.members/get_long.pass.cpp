//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class num_get<charT, InputIterator>

// iter_type get(iter_type in, iter_type end, ios_base&,
//               ios_base::iostate& err, long& v) const;

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <locale>
#include <ios>
#include <cassert>
#include <limits>
#include <streambuf>
#include "test_macros.h"
#include "test_iterators.h"

typedef std::num_get<char, cpp17_input_iterator<const char*> > F;

class my_facet
    : public F
{
public:
    explicit my_facet(std::size_t refs = 0)
        : F(refs) {}
};

class my_numpunct
    : public std::numpunct<char>
{
public:
    my_numpunct() : std::numpunct<char>() {}

protected:
    virtual char_type do_thousands_sep() const {return '_';}
    virtual std::string do_grouping() const {return std::string("\1\2\3");}
};

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    long v = -1;
    const std::ios_base::fmtflags zf = static_cast<std::ios_base::fmtflags>(0);
    {
        const char str[] = "123";
        assert((ios.flags() & ios.basefield) == ios.dec);
        assert(ios.getloc().name() == "C");
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+3);
        assert(err == ios.goodbit);
        assert(v == 123);
    }
    {
        const char str[] = "-123";
        assert((ios.flags() & ios.basefield) == ios.dec);
        assert(ios.getloc().name() == "C");
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+4);
        assert(err == ios.goodbit);
        assert(v == -123);
    }
    {
        const char str[] = "123";
        std::oct(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+3);
        assert(err == ios.goodbit);
        assert(v == 83);
    }
    {
        const char str[] = "123";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+3);
        assert(err == ios.goodbit);
        assert(v == 291);
    }
    {
        const char str[] = "a123";
        std::dec(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str);
        assert(err == ios.failbit);
        assert(v == 0);
    }
    {
        const char str[] = "0x123";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 291);
    }
    {
        const char str[] = "123";
        ios.setf(zf, ios.basefield);
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
        const char str[] = "0x123";
        ios.setf(zf, ios.basefield);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 291);
    }
    {
        const char str[] = "0123";
        ios.setf(zf, ios.basefield);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 83);
    }
    {
        // See PR11871
        const char str[] = "2-";
        ios.setf(zf, ios.basefield);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+1);
        assert(err == ios.goodbit);
        assert(v == 2);
    }
    std::dec(ios);
    ios.imbue(std::locale(std::locale(), new my_numpunct));
    {
        v = -1;
        const char str[] = "123"; // no separators at all
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
        v = -1;
        const char str[] = "+1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+1_";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+_1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "_+1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+1__";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+_1_";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "_+1_";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+__1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "_+_1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "__+1";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1);
    }
    {
        v = -1;
        const char str[] = "+1_2";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 12);
    }
    {
        v = -1;
        const char str[] = "+12_";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 12);
    }
    {
        v = -1;
        const char str[] = "+_12";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 12);
    }
    {
        v = -1;
        const char str[] = "+1__2";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 12);
    }
    {
        v = -1;
        const char str[] = "+12_3";
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
        v = -1;
        const char str[] = "+1_23";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 123);
    }
    {
        v = -1;
        const char str[] = "+1_23_4";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1234);
    }
    {
        v = -1;
        const char str[] = "+123_4";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1234);
    }
    {
        v = -1;
        const char str[] = "+12_34";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1234);
    }
    {
        v = -1;
        const char str[] = "+12_34_5";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 12345);
    }
    {
        v = -1;
        const char str[] = "+123_45_6";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 123456);
    }
    {
        v = -1;
        const char str[] = "+1_23_45_6";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 123456);
    }
    {
        v = -1;
        const char str[] = "+1_234_56_7";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1234567);
    }
    {
        v = -1;
        const char str[] = "+1_234_567_89_0";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1234567890);
    }
    {
        v = -1;
        const char str[] = "-1_234_567_89_0";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == -1234567890);
    }
    {
        v = -1;
        const char str[] = "1_234_567_89_0";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 1234567890);
    }
    {
        v = -1;
        const char str[] = "1234_567_89_0";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == 1234567890);
    }
    {
        v = -1;
        const char str[] = "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_"
                           "1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_1_2_3_4_5_6_7_8_9_0_";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == std::numeric_limits<long>::max());
    }
    {
      v                          = -1;
      const char str[]           = "";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter =
          f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str), ios, err, v);
      assert(base(iter) == str);
      assert(err == (std::ios::eofbit | std::ios::failbit));
      assert(v == 0);
    }
    {
      v                          = -1;
      const char str[]           = "+";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter =
          f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 1), ios, err, v);
      assert(base(iter) == str + 1);
      assert(err == (std::ios::eofbit | std::ios::failbit));
      assert(v == 0);
    }
    {
      v                          = -1;
      const char str[]           = "+";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(std::begin(str)),
          cpp17_input_iterator<const char*>(std::end(str)),
          ios,
          err,
          v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0);
    }
    {
      v                          = -1;
      const char str[]           = "-";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(std::begin(str)),
          cpp17_input_iterator<const char*>(std::end(str)),
          ios,
          err,
          v);
      assert(base(iter) == str + 1);
      assert(err == ios.failbit);
      assert(v == 0);
    }
    {
      v                          = -1;
      const char str[]           = "0";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(std::begin(str)),
          cpp17_input_iterator<const char*>(std::end(str)),
          ios,
          err,
          v);
      assert(base(iter) == str + 1);
      assert(err == ios.goodbit);
      assert(v == 0);
    }
    {
      v                          = -1;
      const char str[]           = "078";
      std::ios_base::iostate err = ios.goodbit;

      ios.flags(ios.flags() & ~ios.basefield);
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(std::begin(str)),
          cpp17_input_iterator<const char*>(std::end(str)),
          ios,
          err,
          v);
      assert(base(iter) == str + 2);
      assert(err == ios.goodbit);
      assert(v == 7);
      ios.flags(ios.flags() | ios.dec);
    }
    {
      v                          = -1;
      std::string str = std::to_string(std::numeric_limits<unsigned long>::max()) + "99a";
      std::ios_base::iostate err = ios.goodbit;

      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str.data()),
          cpp17_input_iterator<const char*>(str.data() + str.size()),
          ios,
          err,
          v);
      assert(base(iter) == str.data() + str.size() - 1);
      assert(err == ios.failbit);
      assert(v == std::numeric_limits<long>::max());
    }
    {
        std::string str = std::to_string(std::numeric_limits<long>::max()) + 'c';
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str.data()),
                  cpp17_input_iterator<const char*>(str.data() + str.size()),
                  ios, err, v);
        assert(base(iter) == str.data() + str.size() - 1);
        assert(err == ios.goodbit);
        assert(v == std::numeric_limits<long>::max());
    }
    {
      std::string str = std::to_string(static_cast<unsigned long>(std::numeric_limits<long>::max()) + 1) + 'c';
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str.data()),
          cpp17_input_iterator<const char*>(str.data() + str.size()),
          ios,
          err,
          v);
      assert(base(iter) == str.data() + str.size() - 1);
      assert(err == ios.failbit);
      assert(v == std::numeric_limits<long>::max());
    }
    {
      std::string str = '-' + std::to_string(static_cast<unsigned long>(std::numeric_limits<long>::max()) + 2) + 'c';
      std::ios_base::iostate err             = ios.goodbit;
      cpp17_input_iterator<const char*> iter = f.get(
          cpp17_input_iterator<const char*>(str.data()),
          cpp17_input_iterator<const char*>(str.data() + str.size()),
          ios,
          err,
          v);
      assert(base(iter) == str.data() + str.size() - 1);
      assert(err == ios.failbit);
      assert(v == std::numeric_limits<long>::min());
    }

  { // Check that auto-detection of the base works properly
    ios.flags(ios.flags() & ~std::ios::basefield);
    { // zeroes
      {
        v                          = -1;
        const char str[]           = "0";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 1), ios, err, v);
        assert(base(iter) == str + 1);
        assert(err == ios.eofbit);
        assert(v == 0);
      }
      {
        v                          = -1;
        const char str[]           = "00";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 2), ios, err, v);
        assert(base(iter) == str + 2);
        assert(err == ios.eofbit);
        assert(v == 0);
      }
      {
        v                          = -1;
        const char str[]           = "0x0";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 3), ios, err, v);
        assert(base(iter) == str + 3);
        assert(err == ios.eofbit);
        assert(v == 0);
      }
      {
        v                          = -1;
        const char str[]           = "0X0";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 3), ios, err, v);
        assert(base(iter) == str + 3);
        assert(err == ios.eofbit);
        assert(v == 0);
      }
    }
    { // first character after base is out of range
      {
        v                          = -1;
        const char str[]           = "08";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 2), ios, err, v);
        assert(base(iter) == str + 1);
        assert(err == ios.goodbit);
        assert(v == 0);
      }
      {
        v                          = -1;
        const char str[]           = "1a";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 2), ios, err, v);
        assert(base(iter) == str + 1);
        assert(err == ios.goodbit);
        assert(v == 1);
      }
      {
        v                          = -1;
        const char str[]           = "0xg";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 3), ios, err, v);
        assert(base(iter) == str + 2);
        assert(err == ios.failbit);
        assert(v == 0);
      }
      {
        v                          = -1;
        const char str[]           = "0Xg";
        std::ios_base::iostate err = ios.goodbit;

        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str), cpp17_input_iterator<const char*>(str + 3), ios, err, v);
        assert(base(iter) == str + 2);
        assert(err == ios.failbit);
        assert(v == 0);
      }
    }
  }

  return 0;
}
