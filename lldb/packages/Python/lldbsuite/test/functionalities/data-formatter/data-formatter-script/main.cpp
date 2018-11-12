//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct i_am_cool
{
	int integer;
	float floating;
	char character;
	i_am_cool(int I, float F, char C) :
    integer(I), floating(F), character(C) {}
	i_am_cool() : integer(1), floating(2), character('3') {}
    
};

struct i_am_cooler
{
	i_am_cool first_cool;
	i_am_cool second_cool;
	float floating;
	
	i_am_cooler(int I1, int I2, float F1, float F2, char C1, char C2) :
    first_cool(I1,F1,C1),
    second_cool(I2,F2,C2),
    floating((F1 + F2)/2) {}
};

int main (int argc, const char * argv[])
{
    i_am_cool one(1,3.14,'E');
    i_am_cool two(4,2.71,'G');
    
    i_am_cool* twoptr = &two;
    
    i_am_cool array[5];
    
    i_am_cooler three(10,4,1985,1/1/2011,'B','E'); // Set break point at this line.
    
    two.integer = 1;
    
    int dummy = 1;
    
    return 0;
}
