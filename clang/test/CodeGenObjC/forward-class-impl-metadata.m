// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface BASE  {
@private
    void* _reserved;
}
@end

@class PVR;

@interface PVRHandldler 
{
          PVR *_imageBrowser;
}
@end

@implementation PVRHandldler @end


@interface PVR   : BASE
@end

@implementation PVR
@end

// Reopen of an interface after use.

@interface A { 
@public 
  int x; 
} 
@property int p0;
@end

int f0(A *a) { 
  return a.p0; 
}

@implementation A
@synthesize p0 = _p0;
@end

@interface B
@end
@class B;
@implementation B
@end
