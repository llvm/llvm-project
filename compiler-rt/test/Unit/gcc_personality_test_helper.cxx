//===-- gcc_personality_test_helper.cxx -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>

extern "C" {
    extern void foo_clean(void* x);
    extern void bar_clean(void* x);
    extern void register_foo_local(int* x);
    extern void register_bar_local(int* x);
    extern void done_foo();
    extern void done_bar();
    extern void foo();
}

static int* foo_x = NULL;
void register_foo_local(int* x)
{
    foo_x = x;
}

static int* bar_x = NULL;
void register_bar_local(int* x)
{
    bar_x = x;
}

static bool foo_clean_called = false;
void foo_clean(void* x)
{
    if  ( foo_x == NULL )
        abort();
    if ( foo_x != (int*)x) 
        abort();
    foo_clean_called = true;
}

static bool bar_clean_called = false;
void bar_clean(void* x)
{
    if  ( bar_x == NULL )
        abort();
    if ( bar_x != (int*)x) 
        abort();
    bar_clean_called = true;
}

void done_foo()
{
}

void done_bar()
{    
    throw "done";
}


//
// foo() is in gcc_personality_test.c and calls bar() which 
// calls done_bar() which throws an exception.
// main() will catch the exception and verify that the cleanup
// routines for foo() and bar() were called by the personality
// function.
//
int main()
{
    try {
        foo();
    }
    catch(...) {
        if ( !foo_clean_called )
            abort();
        if ( !bar_clean_called )
            abort();
		return 0;
    }
	abort();
}
