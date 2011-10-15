//===------------------------ iostream.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "__std_stream"
#include "string"

_LIBCPP_BEGIN_NAMESPACE_STD

static __stdinbuf<char>  __cin(stdin);
static __stdoutbuf<char> __cout(stdout);
static __stdoutbuf<char> __cerr(stderr);
static __stdinbuf<wchar_t>  __wcin(stdin);
static __stdoutbuf<wchar_t> __wcout(stdout);
static __stdoutbuf<wchar_t> __wcerr(stderr);

istream cin(&__cin);
ostream cout(&__cout);
ostream cerr(&__cerr);
ostream clog(&__cerr);
wistream wcin(&__wcin);
wostream wcout(&__wcout);
wostream wcerr(&__wcerr);
wostream wclog(&__wcerr);

ios_base::Init __start_std_streams;

ios_base::Init::Init()
{
    cin.tie(&cout);
    _VSTD::unitbuf(cerr);
    cerr.tie(&cout);

    wcin.tie(&wcout);
    _VSTD::unitbuf(wcerr);
    wcerr.tie(&wcout);
}

ios_base::Init::~Init()
{
    cout.flush();
    clog.flush();

    wcout.flush();
    wclog.flush();
}

_LIBCPP_END_NAMESPACE_STD
