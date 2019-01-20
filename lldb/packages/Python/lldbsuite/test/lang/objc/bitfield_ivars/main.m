//===-- main.m -------------------------------------------*- Objective-C-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

@interface HasBitfield : NSObject {
@public
    unsigned field1 : 1;
    unsigned field2 : 1;
};

-(id)init;
@end

@implementation HasBitfield
-(id)init {
    self = [super init];
    field1 = 0;
    field2 = 1;
    return self;
}
@end

@interface ContainsAHasBitfield : NSObject {
@public
    HasBitfield *hb;
};
-(id)init;
@end

@implementation  ContainsAHasBitfield
-(id)init {
    self = [super init];
    hb = [[HasBitfield alloc] init];
    return self;
}

@end

int main(int argc, const char * argv[]) {
    ContainsAHasBitfield *chb = [[ContainsAHasBitfield alloc] init];
    printf("%d\n", chb->hb->field2); //% self.expect("expression -- chb->hb->field1", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["= 0"])
                                     //% self.expect("expression -- chb->hb->field2", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["= 1"]) # this must happen second
    return 0;
}

