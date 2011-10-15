//===-- clear_cache_test.c - Test clear_cache -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>



extern void __clear_cache(void* start, void* end);

typedef int (*pfunc)(void);

int func1() 
{
    return 1;
}

int func2() 
{
    return 2;
}



unsigned char execution_buffer[128];

int main()
{
    // make executable the page containing execution_buffer 
    char* start = (char*)((uintptr_t)execution_buffer & (-4095));
    char* end = (char*)((uintptr_t)(&execution_buffer[128+4096]) & (-4095));
    if ( mprotect(start, end-start, PROT_READ|PROT_WRITE|PROT_EXEC) != 0 )
        return 1;

    // verify you can copy and execute a function
    memcpy(execution_buffer, (void *)(uintptr_t)&func1, 128);
    __clear_cache(execution_buffer, &execution_buffer[128]);
    pfunc f1 = (pfunc)(uintptr_t)execution_buffer;
    if ( (*f1)() != 1 )
        return 1;

    // verify you can overwrite a function with another
    memcpy(execution_buffer, (void *)(uintptr_t)&func2, 128);
    __clear_cache(execution_buffer, &execution_buffer[128]);
    pfunc f2 = (pfunc)(uintptr_t)execution_buffer;
    if ( (*f2)() != 2 )
        return 1;

    return 0;
}
