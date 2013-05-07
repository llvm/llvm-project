//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class future_error

// const error_code& code() const throw();

#include <future>
#include <cassert>

int main()
{
    {
        std::error_code ec = std::make_error_code(std::future_errc::broken_promise);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::future_already_retrieved);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::promise_already_satisfied);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
    {
        std::error_code ec = std::make_error_code(std::future_errc::no_state);
        std::future_error f(ec);
        assert(f.code() == ec);
    }
}
