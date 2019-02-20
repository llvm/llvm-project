//===-- main.m ---------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#import <Foundation/Foundation.h>

int main() {
    NSDictionary* dic = @{@1 : @2};
    NSLog(@"hello world"); //% dic = self.frame().FindVariable("dic")
    //% dic.SetPreferSyntheticValue(True)
    //% dic.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
    //% dic.SetValueFromCString("12")
    return 0; //% dic = self.frame().FindVariable("dic")
    //% self.assertTrue(dic.GetValueAsUnsigned() == 0xC, "failed to read what I wrote")
}
