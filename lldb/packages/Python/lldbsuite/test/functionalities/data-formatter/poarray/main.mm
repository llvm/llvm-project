//===-- main.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

struct ThreeObjects
{
  id one;
  id two;
  id three;
};

int main()
{
  NSArray *array1 = @[@0xDEADBEEF, @0xFEEDBEEF, @0xBEEFFADE];
  NSArray *array2 = @[@"Hello", @"World"];
  NSDictionary *dictionary = @{@1: array2, @"Two": array2};
  ThreeObjects *tobjects = new ThreeObjects();
  tobjects->one = array1;
  tobjects->two = array2;
  tobjects->three = dictionary;
  id* objects = (id*)tobjects;
  return 0; // break here
}
