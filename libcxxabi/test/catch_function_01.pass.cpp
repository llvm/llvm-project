//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Can you have a catch clause of array type that catches anything?

// GCC incorrectly allows function pointer to be caught by reference.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69372
// XFAIL: gcc
// UNSUPPORTED: no-exceptions

// 65ace9daa360 made it in the dylib in macOS 10.11
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10}}

#include <cassert>

template <class Tp>
bool can_convert(Tp) { return true; }

template <class>
bool can_convert(...) { return false; }

void f() {}

int main(int, char**)
{
    typedef void Function();
    assert(!can_convert<Function&>(&f));
    assert(!can_convert<void*>(&f));
    try
    {
        throw f;     // converts to void (*)()
        assert(false);
    }
    catch (Function& b)  // can't catch void (*)()
    {
        assert(false);
    }
    catch (void*) // can't catch as void*
    {
        assert(false);
    }
    catch(Function*)
    {
    }
    catch (...)
    {
        assert(false);
    }

    return 0;
}
