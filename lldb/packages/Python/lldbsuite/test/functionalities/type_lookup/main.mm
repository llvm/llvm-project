//===-- main.mm -----------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

