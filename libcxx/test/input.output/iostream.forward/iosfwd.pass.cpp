//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iosfwd>

#include <iosfwd>
#include <cwchar>  // for mbstate_t

int main()
{
    {
    std::char_traits<char>*               t1 = 0;
    std::char_traits<wchar_t>*            t2 = 0;
    std::char_traits<unsigned short>*     t3 = 0;
    }
    {
    std::basic_ios<char>*                 t1 = 0;
    std::basic_ios<wchar_t>*              t2 = 0;
    std::basic_ios<unsigned short>*       t3 = 0;
    }
    {
    std::basic_streambuf<char>*           t1 = 0;
    std::basic_streambuf<wchar_t>*        t2 = 0;
    std::basic_streambuf<unsigned short>* t3 = 0;
    }
    {
    std::basic_istream<char>*             t1 = 0;
    std::basic_istream<wchar_t>*          t2 = 0;
    std::basic_istream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_ostream<char>*             t1 = 0;
    std::basic_ostream<wchar_t>*          t2 = 0;
    std::basic_ostream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_iostream<char>*             t1 = 0;
    std::basic_iostream<wchar_t>*          t2 = 0;
    std::basic_iostream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_stringbuf<char>*             t1 = 0;
    std::basic_stringbuf<wchar_t>*          t2 = 0;
    std::basic_stringbuf<unsigned short>*   t3 = 0;
    }
    {
    std::basic_istringstream<char>*             t1 = 0;
    std::basic_istringstream<wchar_t>*          t2 = 0;
    std::basic_istringstream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_ostringstream<char>*             t1 = 0;
    std::basic_ostringstream<wchar_t>*          t2 = 0;
    std::basic_ostringstream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_stringstream<char>*             t1 = 0;
    std::basic_stringstream<wchar_t>*          t2 = 0;
    std::basic_stringstream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_filebuf<char>*             t1 = 0;
    std::basic_filebuf<wchar_t>*          t2 = 0;
    std::basic_filebuf<unsigned short>*   t3 = 0;
    }
    {
    std::basic_ifstream<char>*             t1 = 0;
    std::basic_ifstream<wchar_t>*          t2 = 0;
    std::basic_ifstream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_ofstream<char>*             t1 = 0;
    std::basic_ofstream<wchar_t>*          t2 = 0;
    std::basic_ofstream<unsigned short>*   t3 = 0;
    }
    {
    std::basic_fstream<char>*             t1 = 0;
    std::basic_fstream<wchar_t>*          t2 = 0;
    std::basic_fstream<unsigned short>*   t3 = 0;
    }
    {
    std::istreambuf_iterator<char>*             t1 = 0;
    std::istreambuf_iterator<wchar_t>*          t2 = 0;
    std::istreambuf_iterator<unsigned short>*   t3 = 0;
    }
    {
    std::ostreambuf_iterator<char>*             t1 = 0;
    std::ostreambuf_iterator<wchar_t>*          t2 = 0;
    std::ostreambuf_iterator<unsigned short>*   t3 = 0;
    }
    {
    std::ios*           t1 = 0;
    std::wios*          t2 = 0;
    }
    {
    std::streambuf*        t1 = 0;
    std::istream*          t2 = 0;
    std::ostream*          t3 = 0;
    std::iostream*         t4 = 0;
    }
    {
    std::stringbuf*            t1 = 0;
    std::istringstream*        t2 = 0;
    std::ostringstream*        t3 = 0;
    std::stringstream*         t4 = 0;
    }
    {
    std::filebuf*         t1 = 0;
    std::ifstream*        t2 = 0;
    std::ofstream*        t3 = 0;
    std::fstream*         t4 = 0;
    }
    {
    std::wstreambuf*        t1 = 0;
    std::wistream*          t2 = 0;
    std::wostream*          t3 = 0;
    std::wiostream*         t4 = 0;
    }
    {
    std::wstringbuf*            t1 = 0;
    std::wistringstream*        t2 = 0;
    std::wostringstream*        t3 = 0;
    std::wstringstream*         t4 = 0;
    }
    {
    std::wfilebuf*         t1 = 0;
    std::wifstream*        t2 = 0;
    std::wofstream*        t3 = 0;
    std::wfstream*         t4 = 0;
    }
    {
    std::fpos<std::mbstate_t>*   t1 = 0;
    std::streampos*              t2 = 0;
    std::wstreampos*             t3 = 0;
    }
}
