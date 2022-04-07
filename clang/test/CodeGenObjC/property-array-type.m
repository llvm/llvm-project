// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s
// rdar://15610943

struct _GLKMatrix4
{
    float m[16];
};
typedef struct _GLKMatrix4 GLKMatrix4;

@interface NSObject @end

@interface MyProgram
- (void)setTransform:(float[16])transform;
@end

@interface ViewController
@property (nonatomic, assign) GLKMatrix4 transform;
@end

@implementation ViewController
- (void)viewDidLoad {
  MyProgram *program;
  program.transform = self.transform.m;
}
@end

// CHECK: [[M:%.*]] = getelementptr inbounds %struct._GLKMatrix4, %struct._GLKMatrix4* [[TMP:%.*]], i32 0, i32 0
// CHECK: [[ARRAYDECAY:%.*]] = getelementptr inbounds [16 x float], [16 x float]* [[M]], i64 0, i64 0
// CHECK: [[SIX:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES
// CHECK:  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, float*)*)(i8* noundef [[SEVEN:%.*]], i8* noundef [[SIX]], float* noundef [[ARRAYDECAY]])
