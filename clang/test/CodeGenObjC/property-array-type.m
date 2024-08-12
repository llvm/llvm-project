// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -emit-llvm -o - %s | FileCheck %s

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

// CHECK: [[M:%.*]] = getelementptr inbounds nuw %struct._GLKMatrix4, ptr [[TMP:%.*]], i32 0, i32 0
// CHECK: [[ARRAYDECAY:%.*]] = getelementptr inbounds [16 x float], ptr [[M]], i64 0, i64 0
// CHECK: [[SIX:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
// CHECK:  call void @objc_msgSend(ptr noundef [[SEVEN:%.*]], ptr noundef [[SIX]], ptr noundef [[ARRAYDECAY]])
