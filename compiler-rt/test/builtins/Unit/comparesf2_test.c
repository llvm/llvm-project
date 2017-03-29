// RUN: %clang_builtins %s %librt -o %t && %run %t

//===-- cmpsf2_test.c - Test __cmpsf2 -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests single-precision soft-float comparisons for the compiler-rt
// library.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int __eqsf2(float, float);
int __gesf2(float, float);
int __gtsf2(float, float);
int __lesf2(float, float);
int __ltsf2(float, float);
int __nesf2(float, float);
int __unordsf2(float, float);

struct TestVector {
    float a;
    float b;
    int eqReference;
    int geReference;
    int gtReference;
    int leReference;
    int ltReference;
    int neReference;
    int unReference;
};

int test__cmpsf2(const struct TestVector *vector) {
    
    if (__eqsf2(vector->a, vector->b) != vector->eqReference) {
        printf("error in __eqsf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __eqsf2(vector->a, vector->b),
               vector->eqReference);
        return 1;
    }
    
    if (__gesf2(vector->a, vector->b) != vector->geReference) {
        printf("error in __gesf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __gesf2(vector->a, vector->b),
               vector->geReference);
        return 1;
    }
    
    if (__gtsf2(vector->a, vector->b) != vector->gtReference) {
        printf("error in __gtsf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __gtsf2(vector->a, vector->b),
               vector->gtReference);
        return 1;
    }
    
    if (__lesf2(vector->a, vector->b) != vector->leReference) {
        printf("error in __lesf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __lesf2(vector->a, vector->b),
               vector->leReference);
        return 1;
    }
    
    if (__ltsf2(vector->a, vector->b) != vector->ltReference) {
        printf("error in __ltsf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __ltsf2(vector->a, vector->b),
               vector->ltReference);
        return 1;
    }
    
    if (__nesf2(vector->a, vector->b) != vector->neReference) {
        printf("error in __nesf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __nesf2(vector->a, vector->b),
               vector->neReference);
        return 1;
    }
    
    if (__unordsf2(vector->a, vector->b) != vector->unReference) {
        printf("error in __unordsf2(%a, %a) = %d, expected %d\n",
               vector->a, vector->b,
               __unordsf2(vector->a, vector->b),
               vector->unReference);
        return 1;
    }
    
    return 0;
}

/*
void generateVectors() {
    
    const float arguments[] = {
        __builtin_nanf(""),
        -__builtin_inff(),
        -0x1.fffffep127,
        -0x1.000002p0
        -0x1.000000p0,
        -0x1.fffffep-1f,
        -0x1.000000p-126f,
        -0x0.fffffep-126f,
        -0x0.000002p-126f,
        -0.0,
         0.0,
         0x0.000002p-126f,
         0x0.fffffep-126f,
         0x1.000000p-126f,
         0x1.fffffep-1f,
         0x1.000000p0,
         0x1.000002p0,
         0x1.fffffep127,
         __builtin_inff()
    };
    
    int numArguments = sizeof arguments / sizeof arguments[0];
    
    for (int i=0; i<numArguments; ++i) {
        for (int j=0; j<numArguments; ++j) {
            const float a = arguments[i];
            const float b = arguments[j];
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
    {__builtin_nanf(""),__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-__builtin_inff(),1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1.fffffep+127f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1p0f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1.fffffep-1f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1p-126f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1.fffffcp-127,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x1p-149f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),-0x0p0f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x0p0f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1p-149f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1.fffffcp-127,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1p-126f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1.fffffep-1f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1p0f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1.000002p0f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),0x1.fffffep+127f,1,-1,-1,1,1,1,1},
    {__builtin_nanf(""),__builtin_inff(),1,-1,-1,1,1,1,1},
    {-__builtin_inff(),__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-__builtin_inff(),-__builtin_inff(),0,0,0,0,0,0,0},
    {-__builtin_inff(),-0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-__builtin_inff(),__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffep+127f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1.fffffep+127f,-0x1.fffffep+127f,0,0,0,0,0,0,0},
    {-0x1.fffffep+127f,-0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,-0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,-0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,-0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep+127f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1p0f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1p0f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x1p0f,-0x1p0f,0,0,0,0,0,0,0},
    {-0x1p0f,-0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,-0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,-0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p0f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffep-1f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1.fffffep-1f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x1.fffffep-1f,-0x1p0f,1,1,1,1,1,1,0},
    {-0x1.fffffep-1f,-0x1.fffffep-1f,0,0,0,0,0,0,0},
    {-0x1.fffffep-1f,-0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,-0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffep-1f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1p-126f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1p-126f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x1p-126f,-0x1p0f,1,1,1,1,1,1,0},
    {-0x1p-126f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {-0x1p-126f,-0x1p-126f,0,0,0,0,0,0,0},
    {-0x1p-126f,-0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-126f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1.fffffcp-127,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1.fffffcp-127,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x1.fffffcp-127,-0x1p0f,1,1,1,1,1,1,0},
    {-0x1.fffffcp-127,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {-0x1.fffffcp-127,-0x1p-126f,1,1,1,1,1,1,0},
    {-0x1.fffffcp-127,-0x1.fffffcp-127,0,0,0,0,0,0,0},
    {-0x1.fffffcp-127,-0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1.fffffcp-127,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x1p-149f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1p0f,1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1p-126f,1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {-0x1p-149f,-0x1p-149f,0,0,0,0,0,0,0},
    {-0x1p-149f,-0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x0p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x1p-149f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {-0x0p0f,-__builtin_inff(),1,1,1,1,1,1,0},
    {-0x0p0f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {-0x0p0f,-0x1p0f,1,1,1,1,1,1,0},
    {-0x0p0f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {-0x0p0f,-0x1p-126f,1,1,1,1,1,1,0},
    {-0x0p0f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {-0x0p0f,-0x1p-149f,1,1,1,1,1,1,0},
    {-0x0p0f,-0x0p0f,0,0,0,0,0,0,0},
    {-0x0p0f,0x0p0f,0,0,0,0,0,0,0},
    {-0x0p0f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {-0x0p0f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x0p0f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x0p0f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x0p0f,-0x1p0f,1,1,1,1,1,1,0},
    {0x0p0f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x0p0f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x0p0f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x0p0f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x0p0f,-0x0p0f,0,0,0,0,0,0,0},
    {0x0p0f,0x0p0f,0,0,0,0,0,0,0},
    {0x0p0f,0x1p-149f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x0p0f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1p-149f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1p-149f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1p-149f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1p-149f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1p-149f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1p-149f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1p-149f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1p-149f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1p-149f,0x0p0f,1,1,1,1,1,1,0},
    {0x1p-149f,0x1p-149f,0,0,0,0,0,0,0},
    {0x1p-149f,0x1.fffffcp-127,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-149f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1.fffffcp-127,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1p0f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,-0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffcp-127,0x1.fffffcp-127,0,0,0,0,0,0,0},
    {0x1.fffffcp-127,0x1p-126f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffcp-127,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1p-126f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1p-126f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1p-126f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1p-126f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1p-126f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1p-126f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1p-126f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1p-126f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1p-126f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1p-126f,0x0p0f,1,1,1,1,1,1,0},
    {0x1p-126f,0x1p-149f,1,1,1,1,1,1,0},
    {0x1p-126f,0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1p-126f,0x1p-126f,0,0,0,0,0,0,0},
    {0x1p-126f,0x1.fffffep-1f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-126f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-126f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-126f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1p-126f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffep-1f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1.fffffep-1f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,0x1p-126f,1,1,1,1,1,1,0},
    {0x1.fffffep-1f,0x1.fffffep-1f,0,0,0,0,0,0,0},
    {0x1.fffffep-1f,0x1p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffep-1f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffep-1f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffep-1f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1p0f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1p0f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1p0f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1p0f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1p0f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1p0f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1p0f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1p0f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1p0f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1p0f,0x0p0f,1,1,1,1,1,1,0},
    {0x1p0f,0x1p-149f,1,1,1,1,1,1,0},
    {0x1p0f,0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1p0f,0x1p-126f,1,1,1,1,1,1,0},
    {0x1p0f,0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1p0f,0x1p0f,0,0,0,0,0,0,0},
    {0x1p0f,0x1.000002p0f,-1,-1,-1,-1,-1,-1,0},
    {0x1p0f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1p0f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1.000002p0f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1.000002p0f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1.000002p0f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x0p0f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1p-149f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1p-126f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1p0f,1,1,1,1,1,1,0},
    {0x1.000002p0f,0x1.000002p0f,0,0,0,0,0,0,0},
    {0x1.000002p0f,0x1.fffffep+127f,-1,-1,-1,-1,-1,-1,0},
    {0x1.000002p0f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {0x1.fffffep+127f,__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {0x1.fffffep+127f,-__builtin_inff(),1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1p0f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1p-126f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,-0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x0p0f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1p-149f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1.fffffcp-127,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1p-126f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1.fffffep-1f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1p0f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1.000002p0f,1,1,1,1,1,1,0},
    {0x1.fffffep+127f,0x1.fffffep+127f,0,0,0,0,0,0,0},
    {0x1.fffffep+127f,__builtin_inff(),-1,-1,-1,-1,-1,-1,0},
    {__builtin_inff(),__builtin_nanf(""),1,-1,-1,1,1,1,1},
    {__builtin_inff(),-__builtin_inff(),1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1.fffffep+127f,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1p0f,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1.fffffep-1f,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1p-126f,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1.fffffcp-127,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x1p-149f,1,1,1,1,1,1,0},
    {__builtin_inff(),-0x0p0f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x0p0f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1p-149f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1.fffffcp-127,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1p-126f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1.fffffep-1f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1p0f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1.000002p0f,1,1,1,1,1,1,0},
    {__builtin_inff(),0x1.fffffep+127f,1,1,1,1,1,1,0},
    {__builtin_inff(),__builtin_inff(),0,0,0,0,0,0,0},
};    

int main(int argc, char *argv[]) {
    const int numVectors = sizeof vectors / sizeof vectors[0];
    int i;
    for (i = 0; i<numVectors; ++i) {
        if (test__cmpsf2(&vectors[i])) return 1;
    }
    return 0;
}
