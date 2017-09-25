//===-- main.m --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#import <Cocoa/Cocoa.h>

@interface ThisClassTestsThings : NSObject
@end

@implementation ThisClassTestsThings
- (int)doSomething {

    id s = self;
    NSLog(@"%@",s); //% s = self.frame().FindVariable("s"); s.SetPreferDynamicValue(lldb.eDynamicCanRunTarget)
    //% s_type = s.GetType()
    //% typeClass = s_type.GetTypeClass()
    //% condition = (typeClass == lldb.eTypeClassClass) or (typeClass ==lldb.eTypeClassObjCObject) or (typeClass == lldb.eTypeClassObjCInterface) or (typeClass == lldb.eTypeClassObjCObjectPointer) or (typeClass == lldb.eTypeClassPointer)
    //% self.assertTrue(condition, "s has the wrong TypeClass")
    return 0;
}
- (id)init {
    return (self = [super init]);
}
@end


int main (int argc, char const *argv[])
{
    return [[[ThisClassTestsThings alloc] init] doSomething];
}
