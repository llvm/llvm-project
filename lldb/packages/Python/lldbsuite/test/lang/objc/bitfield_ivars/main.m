//===-- main.m -------------------------------------------*- Objective-C-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

