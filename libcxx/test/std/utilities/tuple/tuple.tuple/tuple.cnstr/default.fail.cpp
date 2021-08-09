//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

#include <tuple>


struct Implicit {
    Implicit() = default;
};

struct Explicit {
    explicit Explicit() = default;
};

std::tuple<> test1() { return {}; }

std::tuple<Implicit> test2() { return {}; }
std::tuple<Explicit> test3() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

std::tuple<Implicit, Implicit> test4() { return {}; }
std::tuple<Explicit, Implicit> test5() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Implicit, Explicit> test6() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Explicit, Explicit> test7() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

std::tuple<Implicit, Implicit, Implicit> test8() { return {}; }
std::tuple<Implicit, Implicit, Explicit> test9() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Implicit, Explicit, Implicit> test10() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Implicit, Explicit, Explicit> test11() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Explicit, Implicit, Implicit> test12() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Explicit, Implicit, Explicit> test13() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Explicit, Explicit, Implicit> test14() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
std::tuple<Explicit, Explicit, Explicit> test15() { return {}; } // expected-error 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**) {
    return 0;
}
