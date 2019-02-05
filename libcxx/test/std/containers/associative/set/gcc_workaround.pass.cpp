//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Tests workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=37804

#include <set>
std::set<int> s;
#include <map>
using std::map;
using std::multimap;

int main(void)
{
       return 0;
}
