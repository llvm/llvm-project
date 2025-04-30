
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#if !defined __DEBUG_PRN_H_INCLUDED__
#define __DEBUG_PRN_H_INCLUDED__ 1

// self-debug
#if defined DO_PRINT
    #include <stdio.h>
    #include <fenv.h>
    #include <string.h>
    #define PRINT(x)    do_printing(&(x), #x, __LINE__, __FILE__, sizeof(x))
#else
    #define PRINT(x)
#endif

#if defined DO_PRINT
static void print_status_flags(char * dst)
{
    if ( fetestexcept(FE_INEXACT) )
    {
        strncat(dst, "P", 1);
    }
    else
    {
        strncat(dst, "p", 1);
    }

    if ( fetestexcept(FE_UNDERFLOW) )
    {
        strncat(dst, "U", 1);
    }
    else
    {
        strncat(dst, "u", 1);
    }

    if ( fetestexcept(FE_OVERFLOW) )   
    {
        strncat(dst, "O", 1);
    }
    else
    {
        strncat(dst, "o", 1);
    }

    if ( fetestexcept(FE_DIVBYZERO) )
    {
        strncat(dst, "Z", 1);
    }
    else
    {
        strncat(dst, "z", 1);
    }
//TODO: add non-Intel compiler support for denormal flag
    {
        strncat(dst, "d", 1);
    }
    if ( fetestexcept(FE_INVALID) )   
    {
        strncat(dst, "I", 1);
    }
    else
    {
        strncat(dst, "i", 1);
    }

    return;
}

static void do_printing(void * pvar, const char * varname, int linenum, const char * filename, int varsize)
{
    char buffer[100];
    char buffer1[150] = "";
    char tmp[16];

    // copy last 15 chars from the file name, pad with spaces, right-justified
    #define THIS_MAX(a, b) ((a) > (b) ? (a) : (b))
    snprintf(buffer, 15+1+1, "%15s:", filename + THIS_MAX((int)strlen(filename) - 15, 0));
    #undef THIS_MAX
    // print line number
    snprintf(tmp, 5+1+1+1, "%5d: ", linenum);
    strncat(buffer, tmp, 5+1+1);
    // print status flags
    print_status_flags(buffer);
    strncat(buffer, ": ", 2);
    // print variable name
    snprintf(tmp, sizeof(tmp), "%14s:", varname);
    strncat(buffer, tmp, sizeof(tmp));
    // print variable value
    switch(varsize)
    {
        case sizeof(float):
        {
            float    fval = *((float*)pvar);
            double   dval = (double)fval;
            unsigned uval = *((unsigned*)pvar);
            snprintf(buffer1, 150, "%s:         %08X == %-20g == %-25a == %d",
                    buffer, uval, dval, fval, uval);
        }
            break;
default:
        case sizeof(double):
        {
            double   dval = *((double*)pvar);
            unsigned long long u64val = *((unsigned long long*)pvar);
            snprintf(buffer1, 150,      "%s: %016llX == %-20g == %-25a == %lld",
                    buffer, u64val, dval, dval, u64val);
        }
            break;
    }

    fprintf(stdout, "%s\n", buffer1);
    return;
}
#endif

#endif //#if !defined __DEBUG_PRN_H_INCLUDED__
