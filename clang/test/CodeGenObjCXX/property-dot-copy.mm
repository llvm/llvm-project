// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://8427922

struct Vector3D
{
		float x, y, z;
		Vector3D();
		Vector3D(const Vector3D &inVector);
		Vector3D(float initX, float initY, float initZ);
		Vector3D &operator=(const Vector3D & rhs);
};

@interface Object3D
{
	Vector3D position;
        Vector3D length;
}
@property (assign) Vector3D position;
- (Vector3D) length;
- (void) setLength: (Vector3D)arg;
@end

int main () 
{
	Object3D *myObject;
        Vector3D V3D(1.0f, 1.0f, 1.0f);
// CHECK: call void @_ZN8Vector3DC1ERKS_
	myObject.position = V3D;

// CHECK: call void @_ZN8Vector3DC1ERKS_
	myObject.length = V3D;

        return 0;
}

// rdar: // 8437253
extern "C" void exit(...);

struct CGPoint {
  float x;
  float y;
};
typedef struct CGPoint CGPoint;

extern "C" const CGPoint CGPointZero;

bool operator==(const CGPoint& a, const CGPoint& b);

@interface TIconViewSettings
@property (assign, nonatomic) CGPoint gridOffset;
@end

@implementation TIconViewSettings
- (CGPoint) gridOffset
{
 return CGPointZero;
}

- (void) foo
{
        if ((self.gridOffset) == CGPointZero)
                exit(1);

 if (self.gridOffset == CGPointZero)
  exit(1);
}
@end

