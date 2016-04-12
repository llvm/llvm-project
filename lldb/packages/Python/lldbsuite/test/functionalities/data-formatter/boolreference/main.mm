//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

int main (int argc, const char * argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	BOOL yes  = YES;
	BOOL no = NO;
    BOOL unset = 12;
	
	BOOL &yes_ref = yes;
	BOOL &no_ref = no;
	BOOL &unset_ref = unset;
	
	BOOL* yes_ptr = &yes;
	BOOL* no_ptr = &no;
	BOOL* unset_ptr = &unset;

    [pool drain];// Set break point at this line.
    return 0;
}

