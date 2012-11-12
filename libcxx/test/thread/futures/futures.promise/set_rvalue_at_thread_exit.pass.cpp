//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class promise<R>

// void promise::set_value_at_thread_exit(R&& r);

#include <future>
#include <memory>
#include <cassert>

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

void func(std::promise<std::unique_ptr<int>> p)
{
    p.set_value_at_thread_exit(std::unique_ptr<int>(new int(5)));
}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
        std::promise<std::unique_ptr<int>> p;
        std::future<std::unique_ptr<int>> f = p.get_future();
        std::thread(func, std::move(p)).detach();
        assert(*f.get() == 5);
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
