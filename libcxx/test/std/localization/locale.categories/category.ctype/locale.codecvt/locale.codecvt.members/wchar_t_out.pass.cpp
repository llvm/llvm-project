//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class codecvt<wchar_t, char, mbstate_t>

// result out(stateT& state,
//            const internT* from, const internT* from_end, const internT*& from_next,
//            externT* to, externT* to_end, externT*& to_next) const;

#include <locale>
#include <string>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstring>

typedef std::codecvt<wchar_t, char, std::mbstate_t> F;

int main()
{
    std::locale l = std::locale::classic();
    const F& f = std::use_facet<F>(l);
    {
        const std::basic_string<F::intern_type> from(L"some text");
        std::vector<char> to(from.size()+1);
        std::mbstate_t mbs = {};
        const F::intern_type* from_next = 0;
        char* to_next = 0;
        F::result r = f.out(mbs, from.data(), from.data() + from.size(), from_next,
                                 to.data(), to.data() + to.size(), to_next);
        assert(r == F::ok);
        assert(static_cast<std::size_t>(from_next - from.data()) == from.size());
        assert(static_cast<std::size_t>(to_next - to.data()) == from.size());
        assert(to.data() == std::string("some text"));
    }
    {
        std::basic_string<F::intern_type> from(L"some text");
        from[4] = '\0';
        std::vector<char> to(from.size()+1);
        std::mbstate_t mbs = {};
        const F::intern_type* from_next = 0;
        char* to_next = 0;
        F::result r = f.out(mbs, from.data(), from.data() + from.size(), from_next,
                                 to.data(), to.data() + to.size(), to_next);
        assert(r == F::ok);
        assert(static_cast<std::size_t>(from_next - from.data()) == from.size());
        assert(static_cast<std::size_t>(to_next - to.data()) == from.size());
        assert(memcmp(to.data(), "some\0text", from.size()) == 0);
    }
    {
        std::basic_string<F::intern_type> from(L"some text");
        std::vector<char> to(from.size()-1);
        std::mbstate_t mbs = {};
        const F::intern_type* from_next = 0;
        char* to_next = 0;
        F::result r = f.out(mbs, from.data(), from.data() + from.size(), from_next,
                                 to.data(), to.data() + to.size()-1, to_next);
        assert(r == F::partial);
        assert(static_cast<std::size_t>(from_next - from.data()) == to.size()-1);
        assert(static_cast<std::size_t>(to_next - to.data()) == to.size()-1);
        assert(to.data() == std::string("some te"));
    }
}
