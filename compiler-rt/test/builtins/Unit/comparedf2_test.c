// RUN: %clang_builtins %s %librt -o %t && %run %t

//===-- cmpdf2_test.c - Test __cmpdf2 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests single-precision soft-double comparisons for the compiler-rt
// library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int __eqdf2(double, double);
int __gedf2(double, double);
int __gtdf2(double, double);
int __ledf2(double, double);
int __ltdf2(double, double);
int __nedf2(double, double);
int __unorddf2(double, double);

struct TestVector {
    double a;
    double b;
    int eqReference;
    int geReference;
    int gtReference;
    int leReference;
    int ltReference;
    int neReference;
    int unReference;
};

int test__cmpdf2(const struct TestVector *vector) {
    
    if (__eqdf2(vector->a, vector->b) != vector->eqReference) {
        printf("error in __eqdf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __eqdf2(vector->a, vector->b),
               vector->eqReference);
        return 1;
    }
    
    if (__gedf2(vector->a, vector->b) != vector->geReference) {
        printf("error in __gedf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __gedf2(vector->a, vector->b),
               vector->geReference);
        return 1;
    }
    
    if (__gtdf2(vector->a, vector->b) != vector->gtReference) {
        printf("error in __gtdf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __gtdf2(vector->a, vector->b),
               vector->gtReference);
        return 1;
    }
    
    if (__ledf2(vector->a, vector->b) != vector->leReference) {
        printf("error in __ledf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __ledf2(vector->a, vector->b),
               vector->leReference);
        return 1;
    }
    
    if (__ltdf2(vector->a, vector->b) != vector->ltReference) {
        printf("error in __ltdf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __ltdf2(vector->a, vector->b),
               vector->ltReference);
        return 1;
    }
    
    if (__nedf2(vector->a, vector->b) != vector->neReference) {
        printf("error in __nedf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __nedf2(vector->a, vector->b),
               vector->neReference);
        return 1;
    }
    
    if (__unorddf2(vector->a, vector->b) != vector->unReference) {
        printf("error in __unorddf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __unorddf2(vector->a, vector->b),
               vector->unReference);
        return 1;
    }
    
    return 0;
}

/*
void generateVectors() {
    
    const double arguments[] = {
        __builtin_nan(""),
        -__builtin_inf(),
        -0x1.fffffffffffffp1023,
        -0x1.0000000000001p0
        -0x1.0000000000000p0,
        -0x1.fffffffffffffp-1,
        -0x1.0000000000000p-1022,
        -0x0.fffffffffffffp-1022,
        -0x0.0000000000001p-1022,
        -0.0,
         0.0,
         0x0.0000000000001p-1022,
         0x0.fffffffffffffp-1022,
         0x1.0000000000000p-1022,
         0x1.fffffffffffffp-1,
         0x1.0000000000000p0,
         0x1.0000000000001p0,
         0x1.fffffffffffffp1023,
         __builtin_inf()
    };
    
    int numArguments = sizeof arguments / sizeof arguments[0];
    
    for (int i=0; i<numArguments; ++i) {
        for (int j=0; j<numArguments; ++j) {
            const double a = arguments[i];
            const double b = arguments[j];
            const int leResult = a < b ? -1 : a == b ? 0 :  1;
            const int geResult = a > b ?  1 : a == b ? 0 : -1;
            const int unResult = a != a || b != b ? 1 : 0;
            printf("{%a,%a,%d,%d,%d,%d,%d,%d,%d},\n",
                   a, b, 
                   leResult,
                   geResult,
                   geResult,
                   leResult,
                   leResult,
                   leResult,
                   unResult);
        }
    }
} */

static const struct TestVector vectors[] = {
    {__builtin_nan(""),__builtin_nan(""),1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-__builtin_inf(),1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1.fffffffffffffp+1023,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1p+1,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1.fffffffffffffp-1,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1p-1022,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1.fffffcp-1023,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x1p-1074,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),-0x0p+0,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x0p+0,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1p-1074,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1.fffffcp-1023,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1p-1022,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1.fffffffffffffp-1,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1p+0,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1.0000000000001p+0,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),0x1.fffffffffffffp+1023,1,-1,-1,1,1,1,1},
    {__builtin_nan(""),__builtin_inf(),1,-1,-1,1,1,1,1},
    {-__builtin_inf(),__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-__builtin_inf(),-__builtin_inf(),0,0,0,0,0,0,0},
    {-__builtin_inf(),-0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x1p+1,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inf(),__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffffffffffp+1023,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1.fffffffffffffp+1023,-0x1.fffffffffffffp+1023,0,0,0,0,0,0,0},
    {-0x1.fffffffffffffp+1023,-0x1p+1,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,-0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,-0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,-0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp+1023,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1p+1,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1p+1,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x1p+1,-0x1p+1,0,0,0,0,0,0,0},
    {-0x1p+1,-0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,-0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,-0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p+1,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffffffffffp-1,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1.fffffffffffffp-1,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x1.fffffffffffffp-1,-0x1p+1,1,1,1,1,1,1,0},
    {-0x1.fffffffffffffp-1,-0x1.fffffffffffffp-1,0,0,0,0,0,0,0},
    {-0x1.fffffffffffffp-1,-0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,-0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffffffffffp-1,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1p-1022,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1p-1022,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x1p-1022,-0x1p+1,1,1,1,1,1,1,0},
    {-0x1p-1022,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {-0x1p-1022,-0x1p-1022,0,0,0,0,0,0,0},
    {-0x1p-1022,-0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1022,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffcp-1023,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1.fffffcp-1023,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x1.fffffcp-1023,-0x1p+1,1,1,1,1,1,1,0},
    {-0x1.fffffcp-1023,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {-0x1.fffffcp-1023,-0x1p-1022,1,1,1,1,1,1,0},
    {-0x1.fffffcp-1023,-0x1.fffffcp-1023,0,0,0,0,0,0,0},
    {-0x1.fffffcp-1023,-0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-1023,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x1p-1074,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1p+1,1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1p-1022,1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {-0x1p-1074,-0x1p-1074,0,0,0,0,0,0,0},
    {-0x1p-1074,-0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x0p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-1074,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {-0x0p+0,-__builtin_inf(),1,1,1,1,1,1,0},
    {-0x0p+0,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {-0x0p+0,-0x1p+1,1,1,1,1,1,1,0},
    {-0x0p+0,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {-0x0p+0,-0x1p-1022,1,1,1,1,1,1,0},
    {-0x0p+0,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {-0x0p+0,-0x1p-1074,1,1,1,1,1,1,0},
    {-0x0p+0,-0x0p+0,0,0,0,0,0,0,0},
    {-0x0p+0,0x0p+0,0,0,0,0,0,0,0},
    {-0x0p+0,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {-0x0p+0,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x0p+0,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x0p+0,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x0p+0,-0x1p+1,1,1,1,1,1,1,0},
    {0x0p+0,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x0p+0,-0x1p-1022,1,1,1,1,1,1,0},
    {0x0p+0,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x0p+0,-0x1p-1074,1,1,1,1,1,1,0},
    {0x0p+0,-0x0p+0,0,0,0,0,0,0,0},
    {0x0p+0,0x0p+0,0,0,0,0,0,0,0},
    {0x0p+0,0x1p-1074,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x0p+0,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1p-1074,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1p-1074,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1p-1074,-0x1p+1,1,1,1,1,1,1,0},
    {0x1p-1074,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1p-1074,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1p-1074,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1p-1074,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1p-1074,-0x0p+0,1,1,1,1,1,1,0},
    {0x1p-1074,0x0p+0,1,1,1,1,1,1,0},
    {0x1p-1074,0x1p-1074,0,0,0,0,0,0,0},
    {0x1p-1074,0x1.fffffcp-1023,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1074,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1.fffffcp-1023,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1p+1,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,-0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffcp-1023,0x1.fffffcp-1023,0,0,0,0,0,0,0},
    {0x1.fffffcp-1023,0x1p-1022,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-1023,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1p-1022,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1p-1022,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1p-1022,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1p-1022,-0x1p+1,1,1,1,1,1,1,0},
    {0x1p-1022,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1p-1022,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1p-1022,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1p-1022,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1p-1022,-0x0p+0,1,1,1,1,1,1,0},
    {0x1p-1022,0x0p+0,1,1,1,1,1,1,0},
    {0x1p-1022,0x1p-1074,1,1,1,1,1,1,0},
    {0x1p-1022,0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1p-1022,0x1p-1022,0,0,0,0,0,0,0},
    {0x1p-1022,0x1.fffffffffffffp-1,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1022,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1022,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1022,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1p-1022,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffffffffffp-1,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1.fffffffffffffp-1,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1p+1,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,-0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,0x1p-1022,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp-1,0x1.fffffffffffffp-1,0,0,0,0,0,0,0},
    {0x1.fffffffffffffp-1,0x1p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffffffffffp-1,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffffffffffp-1,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffffffffffp-1,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1p+0,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1p+0,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1p+0,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1p+0,-0x1p+1,1,1,1,1,1,1,0},
    {0x1p+0,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1p+0,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1p+0,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1p+0,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1p+0,-0x0p+0,1,1,1,1,1,1,0},
    {0x1p+0,0x0p+0,1,1,1,1,1,1,0},
    {0x1p+0,0x1p-1074,1,1,1,1,1,1,0},
    {0x1p+0,0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1p+0,0x1p-1022,1,1,1,1,1,1,0},
    {0x1p+0,0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1p+0,0x1p+0,0,0,0,0,0,0,0},
    {0x1p+0,0x1.0000000000001p+0,-1,-1,-1,-1,-1,-1,0},
    {0x1p+0,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1p+0,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1.0000000000001p+0,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1.0000000000001p+0,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1p+1,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,-0x0p+0,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x0p+0,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1p-1074,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1p-1022,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1p+0,1,1,1,1,1,1,0},
    {0x1.0000000000001p+0,0x1.0000000000001p+0,0,0,0,0,0,0,0},
    {0x1.0000000000001p+0,0x1.fffffffffffffp+1023,-1,-1,-1,-1,-1,-1,0},
    {0x1.0000000000001p+0,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffffffffffp+1023,__builtin_nan(""),1,-1,-1,1,1,1,1},
    {0x1.fffffffffffffp+1023,-__builtin_inf(),1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1p+1,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1p-1022,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,-0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x0p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1p-1074,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1p-1022,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1.0000000000001p+0,1,1,1,1,1,1,0},
    {0x1.fffffffffffffp+1023,0x1.fffffffffffffp+1023,0,0,0,0,0,0,0},
    {0x1.fffffffffffffp+1023,__builtin_inf(),-1,-1,-1,-1,-1,-1,0},
    {__builtin_inf(),__builtin_nan(""),1,-1,-1,1,1,1,1},
    {__builtin_inf(),-__builtin_inf(),1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1p+1,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1p-1022,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x1p-1074,1,1,1,1,1,1,0},
    {__builtin_inf(),-0x0p+0,1,1,1,1,1,1,0},
    {__builtin_inf(),0x0p+0,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1p-1074,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1.fffffcp-1023,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1p-1022,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1.fffffffffffffp-1,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1p+0,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1.0000000000001p+0,1,1,1,1,1,1,0},
    {__builtin_inf(),0x1.fffffffffffffp+1023,1,1,1,1,1,1,0},
    {__builtin_inf(),__builtin_inf(),0,0,0,0,0,0,0},
};    

int main(int argc, char *argv[]) {
    const int numVectors = sizeof vectors / sizeof vectors[0];
    int i;
    for (i = 0; i<numVectors; ++i) {
        if (test__cmpdf2(&vectors[i])) return 1;
    }
    return 0;
}
