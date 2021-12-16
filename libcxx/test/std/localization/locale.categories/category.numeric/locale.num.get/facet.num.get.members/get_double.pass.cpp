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
//               ios_base::iostate& err, double& v) const;

#include <locale>
#include <ios>
#include <cassert>
#include <streambuf>
#include <cmath>
#include "test_macros.h"
#include "test_iterators.h"
#include "hexfloat.h"
#include "get_float_common.h"

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
    virtual char_type do_decimal_point() const {return ';';}
    virtual char_type do_thousands_sep() const {return '_';}
    virtual std::string do_grouping() const {return std::string("\1\2\3");}
};

int main(int, char**)
{
    const my_facet f(1);
    std::ios ios(0);
    double v = -1;

#ifdef _LIBCPP_ABI_LOCAL_NUM_GET_NEW_STAGE2_FLOAT_LOOP
    // Valid floating point formats where whole string is consumed
    TEST("0x123.4f", 8, (double)hexfloat<double>(0x123, 0x4f, 0), ios.eofbit);
    TEST("inf", 3, INFINITY, ios.goodbit | ios.eofbit);
    TEST("Inf", 3, INFINITY, ios.goodbit | ios.eofbit);
    TEST("+iNf", 4, INFINITY, ios.goodbit | ios.eofbit);
    TEST("-inF", 4, -INFINITY, ios.goodbit | ios.eofbit);
    TEST("INFxyz", 3, INFINITY, ios.goodbit);

    // Valid floating point formats with unparsed trailing characters
    TEST("123.4f", 5, 123.4, ios.goodbit);
    TEST("123xyz", 3, 123.0, ios.goodbit);
    TEST("0x123.4+", 7, (double)hexfloat<double>(0x123, 0x4, 0), ios.goodbit);

    // Shouldn't recognise e, p or x more than once
    TEST("123.4e-5e-4", 8, 123.4e-5, ios.goodbit);
    TEST("0x123.4p-5p-4", 10, (double)hexfloat<double>(0x123, 0x4, -5), ios.goodbit);
    TEST("0x123x5", 5, (double)hexfloat<double>(0x123, 0x0, 0), ios.goodbit);

    // Invalid (non-float) inputs
    TEST("a", 0, 0.0, ios.failbit);
    TEST("e", 0, 0.0, ios.failbit);
    TEST("f", 0, 0.0, ios.failbit);
    TEST("p", 0, 0.0, ios.failbit);
    TEST("M", 0, 0.0, ios.failbit);
    TEST("{}", 0, 0.0, ios.failbit);
    TEST("x123", 0, 0.0, ios.failbit);

    // Incomplete inputs, i.e. eof before finished parsing
    TEST("-", 1, 0.0, ios.eofbit | ios.failbit);
    TEST("+", 1, 0.0, ios.eofbit | ios.failbit);
    TEST("0x123.4p", 8, 0.0, ios.eofbit | ios.failbit);
#endif

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
        assert(v == hexfloat<double>(0x125, 0, -1));
    }
    {
        const char str[] = "inf";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "INF";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == INFINITY);
    }
    {
        const char str[] = "-inf";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == -INFINITY);
    }
    {
        const char str[] = "-INF";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == -INFINITY);
    }
    {
        const char str[] = "nan";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(std::isnan(v));
    }
    {
        const char str[] = "NAN";
        std::hex(ios);
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(std::isnan(v));
    }
    {
        v = -1;
        const char str[] = "123_456_78_9;125";
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
        // See PR11871
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
        v = -1;
        const char str[] = "1.79779e+309"; // unrepresentable
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == HUGE_VAL);
    }
    {
        v = -1;
        const char str[] = "-1.79779e+308"; // unrepresentable
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.failbit);
        assert(v == -HUGE_VAL);
    }
    ios.imbue(std::locale(std::locale(), new my_numpunct));
    {
        v = -1;
        const char str[] = "123_456_78_9;125";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(v == 123456789.125);
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
    }
    {
        // See PR15445
        v = -1;
        const char str[] = "3;14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651e+10";
        std::ios_base::iostate err = ios.goodbit;
        cpp17_input_iterator<const char*> iter =
            f.get(cpp17_input_iterator<const char*>(str),
                  cpp17_input_iterator<const char*>(str+sizeof(str)),
                  ios, err, v);
        assert(base(iter) == str+sizeof(str)-1);
        assert(err == ios.goodbit);
        assert(std::abs(v - 3.14159265358979e+10)/3.14159265358979e+10 < 1.e-8);
    }

  return 0;
}
