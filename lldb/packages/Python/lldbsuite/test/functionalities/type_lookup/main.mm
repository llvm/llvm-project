//===-- main.mm -----------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

class MyCPPClass {
public:
    MyCPPClass(float f) : f(f) {}
    
    float setF(float f) {
      float oldf = this->f;
      this->f = f;
      return oldf;
    }
    
    float getF() {
      return f;
    }
private:
    float f;
};

typedef MyCPPClass MyClass;

@interface MyObjCClass : NSObject {
  int x;
}
- (id)init;
- (int)read;
@end

@implementation MyObjCClass {
  int y;
}
- (id)init {
  if (self = [super init]) {
    self->x = 12;
    self->y = 24;
  }
  return self;
}
- (int)read {
  return self->x + self->y;
}
@end

int main (int argc, const char * argv[])
{
  MyClass my_cpp(3.1415);
  return 0; // break here
}

