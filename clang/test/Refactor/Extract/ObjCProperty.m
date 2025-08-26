// RUN: clang-refactor extract -selection=test:%s %s -- 2>&1 | grep -v CHECK | FileCheck %s

@interface HasProperty

@property (strong) HasProperty *item;

- (HasProperty *)implicitProp;

- (void)setImplicitSetter:(HasProperty *)value;

@end

@implementation HasProperty

- (void)allowGetterExtraction {
  /*range allow_getter=->+0:42*/self.item;
  /*range allow_imp_getter=->+0:54*/self.implicitProp;
}
// CHECK: 1 'allow_getter' results:
// CHECK:      extracted() {
// CHECK-NEXT: return self.item;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: - (void)allowGetterExtraction {
// CHECK-NEXT: extracted();

// CHECK: 1 'allow_imp_getter' results:
// CHECK:      extracted() {
// CHECK-NEXT: return self.implicitProp;{{$}}
// CHECK-NEXT: }{{[[:space:]].*}}
// CHECK-NEXT: - (void)allowGetterExtraction {
// CHECK-NEXT: self.item;
// CHECK-NEXT: extracted();

- (void)prohibitSetterExtraction {
  /*range prohibit_setter=->+0:45*/self.item = 0;
  /*range prohibit_setter=->+0:55*/self.implicitSetter = 0;
}
// CHECK: 2 'prohibit_setter' results:
// CHECK: the selected expression cannot be extracted

@end
