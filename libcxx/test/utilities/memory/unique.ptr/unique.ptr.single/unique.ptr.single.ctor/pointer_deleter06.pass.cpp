//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr(pointer, deleter) ctor

#include <memory>
#include <cassert>

// unique_ptr(pointer, deleter) should work with function pointers
// unique_ptr<void> should work

bool my_free_called = false;

void my_free(void*)
{
    my_free_called = true;
}

int main()
{
    {
    int i = 0;
    std::unique_ptr<void, void (*)(void*)> s(&i, my_free);
    assert(s.get() == &i);
    assert(s.get_deleter() == my_free);
    assert(!my_free_called);
    }
    assert(my_free_called);
}
