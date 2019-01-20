//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

void foo(int a, int b)
{
    printf("%d %d\n", a, b);
}

void bar(int *ptr)
{
	printf("%d\n", *ptr);
}

int main (int argc, const char * argv[])
{
    foo(42, 56);
    int i = 78;
    bar(&i);
    return 0;
}
