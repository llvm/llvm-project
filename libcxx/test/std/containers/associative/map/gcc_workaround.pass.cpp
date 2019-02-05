//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Tests workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=37804

#include <map>
std::map<int,int>::iterator it;
#include <set>
using std::set;
using std::multiset;

int main(void)
{
       return 0;
}
