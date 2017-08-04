//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test that <bitset> includes <cstddef>, <string>, <stdexcept> and <iosfwd>

#include <bitset>

template <class> void test_typedef() {}

int main()
{
  { // test for <cstddef>
    std::ptrdiff_t p; ((void)p);
    std::size_t s; ((void)s);
    std::nullptr_t np; ((void)np);
  }
  { // test for <string>
    std::string s; ((void)s);
  }
  { // test for <stdexcept>
    std::logic_error le("blah"); ((void)le);
    std::runtime_error re("blah"); ((void)re);
  }
  { // test for <iosfwd>
    test_typedef<std::ios>();
    test_typedef<std::wios>();
    test_typedef<std::istream>();
    test_typedef<std::ostream>();
    test_typedef<std::iostream>();
  }
}
