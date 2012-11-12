//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <istream>

// basic_istream<charT,traits>& seekg(off_type off, ios_base::seekdir dir);

#include <istream>
#include <cassert>

int seekoff_called = 0;

template <class CharT>
struct testbuf
    : public std::basic_streambuf<CharT>
{
    typedef std::basic_string<CharT> string_type;
    typedef std::basic_streambuf<CharT> base;
private:
    string_type str_;
public:

    testbuf() {}
    testbuf(const string_type& str)
        : str_(str)
    {
        base::setg(const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()),
                   const_cast<CharT*>(str_.data()) + str_.size());
    }

    CharT* eback() const {return base::eback();}
    CharT* gptr() const {return base::gptr();}
    CharT* egptr() const {return base::egptr();}
protected:
    typename base::pos_type seekoff(typename base::off_type off,
                                    std::ios_base::seekdir way,
                                    std::ios_base::openmode which)
    {
        assert(which == std::ios_base::in);
        ++seekoff_called;
        return off;
    }
};

int main()
{
    {
        testbuf<char> sb(" 123456789");
        std::istream is(&sb);
        is.seekg(5, std::ios_base::cur);
        assert(is.good());
        assert(seekoff_called == 1);
    }
    {
        testbuf<wchar_t> sb(L" 123456789");
        std::wistream is(&sb);
        is.seekg(5, std::ios_base::cur);
        assert(is.good());
        assert(seekoff_called == 2);
    }
}
