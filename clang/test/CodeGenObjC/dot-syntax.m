// RUN: %clang_cc1 -emit-llvm -o %t %s

int printf(const char *, ...);

@interface Root
-(id) alloc;
-(id) init;
@end

typedef struct {
  float x, y, z[2];
} S;

@interface A : Root {
  int myX;
  //  __complex myY;
  S myZ;
}

@property int x;
//@property __complex int y;
@property S z;
@end

@implementation A
-(int) x {
  printf("-[A x] = %d\n", myX);
  return myX;
}
-(void) setX: (int) arg {
  myX = arg;
  printf("-[A setX: %d]\n", myX);
}

// FIXME: Add back
#if 0
-(__complex int) y {
  printf("-[A y] = (%d, %d)\n", __real myY, __imag myY);
  return myY;
}
-(void) setY: (__complex int) arg {
  myY = arg;
  printf("-[A setY: (%d, %d)]\n", __real myY, __imag myY);
}
#endif

-(S) z {
  printf("-[A z] = { %f, %f, { %f, %f } }\n", 
         myZ.x, myZ.y, myZ.z[0], myZ.z[1]);
  return myZ;
}
-(void) setZ: (S) arg {
  myZ = arg;
  printf("-[A setZ: { %f, %f, { %f, %f } } ]\n", 
         myZ.x, myZ.y, myZ.z[0], myZ.z[1]);
}

@end

int main() {
#define SWAP(T,a,b) { T a_tmp = a; a = b; b = a_tmp; }
  A *a = [[A alloc] init];
  A *b = [[A alloc] init];
  int a0 = 23;
  //  __complex a1 = 25 + 10i;
  S a2 =  { 246, 458, {275, 12} };
  int b0 = 42673;
  //  __complex b1 = 15 + 13i;
  S b2 =  { 26, 2, {367, 13} };

  a.x = a0;
  //  a.y = a1;
  a.z = a2;

  a.x += a0;
  //  a.y += a1;
  // Yay, no compound assign of structures. A GCC extension in the
  // works, perhaps?

  b.x = b0;
  //  b.y = b1;
  b.z = b2;

  int x0 = (b.x = b0);
  printf("(b.x = b0): %d\n", x0);

  //  int x1 = __real (b.y = b1);
  //  printf("__real (b.y = b1) = %d\n", x1);

  float x2 = (b.z = b2).x;
  printf("(b.z = b2).x: %f\n", x2);

  SWAP(int, a.x, b.x);
  //  SWAP(__complex int, a.y, b.y);
  SWAP(S, a.z, b.z);

  return 0;
}
