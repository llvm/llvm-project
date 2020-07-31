//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure functions specified as being 'addressable' (their address can be
// taken in a well-defined manner) are indeed addressable. This notion was
// added by http://wg21.link/p0551. While it was technically only introduced
// in C++20, we test it in all standard modes because it's basic QOI to provide
// a consistent behavior for that across standard modes.

// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %t.tu1.o -DTU1
// RUN: %{cxx} %{flags} %{compile_flags} -c %s -o %t.tu2.o -DTU2
// RUN: %{cxx} %{flags} %{link_flags} %t.tu1.o %t.tu2.o -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <utility>


typedef std::ios_base& (FormatFlagFunction)(std::ios_base&);
typedef std::basic_ostream<char>& (OstreamManipFunction)(std::basic_ostream<char>&);
typedef std::basic_ostream<wchar_t>& (WOstreamManipFunction)(std::basic_ostream<wchar_t>&);
typedef std::basic_istream<char>& (IstreamManipFunction)(std::basic_istream<char>&);
typedef std::basic_istream<wchar_t>& (WIstreamManipFunction)(std::basic_istream<wchar_t>&);

extern FormatFlagFunction* get_formatflag_tu1(std::string);
extern FormatFlagFunction* get_formatflag_tu2(std::string);

extern OstreamManipFunction* get_ostreammanip_tu1(std::string);
extern OstreamManipFunction* get_ostreammanip_tu2(std::string);
extern WOstreamManipFunction* get_wostreammanip_tu1(std::string);
extern WOstreamManipFunction* get_wostreammanip_tu2(std::string);

extern IstreamManipFunction* get_istreammanip_tu1(std::string);
extern IstreamManipFunction* get_istreammanip_tu2(std::string);
extern WIstreamManipFunction* get_wistreammanip_tu1(std::string);
extern WIstreamManipFunction* get_wistreammanip_tu2(std::string);

#ifdef TU1
FormatFlagFunction* get_formatflag_tu1(std::string func)
#else
FormatFlagFunction* get_formatflag_tu2(std::string func)
#endif
{
    std::map<std::string, FormatFlagFunction*> all_funcs;

    // [fmtflags.manip]
    all_funcs.insert(std::make_pair("boolalpha", &std::boolalpha));
    all_funcs.insert(std::make_pair("noboolalpha", &std::noboolalpha));
    all_funcs.insert(std::make_pair("showbase", &std::showbase));
    all_funcs.insert(std::make_pair("noshowbase", &std::noshowbase));
    all_funcs.insert(std::make_pair("showpoint", &std::showpoint));
    all_funcs.insert(std::make_pair("noshowpoint", &std::noshowpoint));
    all_funcs.insert(std::make_pair("showpos", &std::showpos));
    all_funcs.insert(std::make_pair("noshowpos", &std::noshowpos));
    all_funcs.insert(std::make_pair("skipws", &std::skipws));
    all_funcs.insert(std::make_pair("noskipws", &std::noskipws));
    all_funcs.insert(std::make_pair("uppercase", &std::uppercase));
    all_funcs.insert(std::make_pair("nouppercase", &std::nouppercase));
    all_funcs.insert(std::make_pair("unitbuf", &std::unitbuf));
    all_funcs.insert(std::make_pair("nounitbuf", &std::nounitbuf));

    // [adjustfield.manip]
    all_funcs.insert(std::make_pair("internal", &std::internal));
    all_funcs.insert(std::make_pair("left", &std::left));
    all_funcs.insert(std::make_pair("right", &std::right));

    // [basefield.manip]
    all_funcs.insert(std::make_pair("dec", &std::dec));
    all_funcs.insert(std::make_pair("hex", &std::hex));
    all_funcs.insert(std::make_pair("oct", &std::oct));

    // [floatfield.manip]
    all_funcs.insert(std::make_pair("fixed", &std::fixed));
    all_funcs.insert(std::make_pair("scientific", &std::scientific));
    all_funcs.insert(std::make_pair("hexfloat", &std::hexfloat));
    all_funcs.insert(std::make_pair("defaultfloat", &std::defaultfloat));

    return all_funcs.at(func);
}

// [ostream.manip] (char)
#ifdef TU1
OstreamManipFunction* get_ostreammanip_tu1(std::string func)
#else
OstreamManipFunction* get_ostreammanip_tu2(std::string func)
#endif
{
    std::map<std::string, OstreamManipFunction*> all_funcs;
    typedef std::char_traits<char> Traits;
    all_funcs.insert(std::make_pair("endl", &std::endl<char, Traits>));
    all_funcs.insert(std::make_pair("ends", &std::ends<char, Traits>));
    all_funcs.insert(std::make_pair("flush", &std::flush<char, Traits>));
    return all_funcs.at(func);
}

// [ostream.manip] (wchar_t)
#ifdef TU1
WOstreamManipFunction* get_wostreammanip_tu1(std::string func)
#else
WOstreamManipFunction* get_wostreammanip_tu2(std::string func)
#endif
{
    std::map<std::string, WOstreamManipFunction*> all_funcs;
    typedef std::char_traits<wchar_t> Traits;
    all_funcs.insert(std::make_pair("endl", &std::endl<wchar_t, Traits>));
    all_funcs.insert(std::make_pair("ends", &std::ends<wchar_t, Traits>));
    all_funcs.insert(std::make_pair("flush", &std::flush<wchar_t, Traits>));
    return all_funcs.at(func);
}

// [istream.manip] (char)
#ifdef TU1
IstreamManipFunction* get_istreammanip_tu1(std::string func)
#else
IstreamManipFunction* get_istreammanip_tu2(std::string func)
#endif
{
    std::map<std::string, IstreamManipFunction*> all_funcs;
    typedef std::char_traits<char> Traits;
    all_funcs.insert(std::make_pair("ws", &std::ws<char, Traits>));
    return all_funcs.at(func);
}

// [istream.manip] (wchar_t)
#ifdef TU1
WIstreamManipFunction* get_wistreammanip_tu1(std::string func)
#else
WIstreamManipFunction* get_wistreammanip_tu2(std::string func)
#endif
{
    std::map<std::string, WIstreamManipFunction*> all_funcs;
    typedef std::char_traits<wchar_t> Traits;
    all_funcs.insert(std::make_pair("ws", &std::ws<wchar_t, Traits>));
    return all_funcs.at(func);
}


#ifdef TU2
    int main() {
        assert(get_formatflag_tu1("boolalpha") == get_formatflag_tu2("boolalpha"));
        assert(get_formatflag_tu1("noboolalpha") == get_formatflag_tu2("noboolalpha"));
        assert(get_formatflag_tu1("showbase") == get_formatflag_tu2("showbase"));
        assert(get_formatflag_tu1("noshowbase") == get_formatflag_tu2("noshowbase"));
        assert(get_formatflag_tu1("showpoint") == get_formatflag_tu2("showpoint"));
        assert(get_formatflag_tu1("noshowpoint") == get_formatflag_tu2("noshowpoint"));
        assert(get_formatflag_tu1("showpos") == get_formatflag_tu2("showpos"));
        assert(get_formatflag_tu1("noshowpos") == get_formatflag_tu2("noshowpos"));
        assert(get_formatflag_tu1("skipws") == get_formatflag_tu2("skipws"));
        assert(get_formatflag_tu1("noskipws") == get_formatflag_tu2("noskipws"));
        assert(get_formatflag_tu1("uppercase") == get_formatflag_tu2("uppercase"));
        assert(get_formatflag_tu1("nouppercase") == get_formatflag_tu2("nouppercase"));
        assert(get_formatflag_tu1("unitbuf") == get_formatflag_tu2("unitbuf"));
        assert(get_formatflag_tu1("nounitbuf") == get_formatflag_tu2("nounitbuf"));
        assert(get_formatflag_tu1("internal") == get_formatflag_tu2("internal"));
        assert(get_formatflag_tu1("left") == get_formatflag_tu2("left"));
        assert(get_formatflag_tu1("right") == get_formatflag_tu2("right"));
        assert(get_formatflag_tu1("dec") == get_formatflag_tu2("dec"));
        assert(get_formatflag_tu1("hex") == get_formatflag_tu2("hex"));
        assert(get_formatflag_tu1("oct") == get_formatflag_tu2("oct"));
        assert(get_formatflag_tu1("fixed") == get_formatflag_tu2("fixed"));
        assert(get_formatflag_tu1("scientific") == get_formatflag_tu2("scientific"));
        assert(get_formatflag_tu1("hexfloat") == get_formatflag_tu2("hexfloat"));
        assert(get_formatflag_tu1("defaultfloat") == get_formatflag_tu2("defaultfloat"));

        assert(get_ostreammanip_tu1("endl") == get_ostreammanip_tu2("endl"));
        assert(get_ostreammanip_tu1("ends") == get_ostreammanip_tu2("ends"));
        assert(get_ostreammanip_tu1("flush") == get_ostreammanip_tu2("flush"));

        assert(get_wostreammanip_tu1("endl") == get_wostreammanip_tu2("endl"));
        assert(get_wostreammanip_tu1("ends") == get_wostreammanip_tu2("ends"));
        assert(get_wostreammanip_tu1("flush") == get_wostreammanip_tu2("flush"));

        assert(get_istreammanip_tu1("ws") == get_istreammanip_tu2("ws"));

        assert(get_wistreammanip_tu1("ws") == get_wistreammanip_tu2("ws"));
    }
#endif
