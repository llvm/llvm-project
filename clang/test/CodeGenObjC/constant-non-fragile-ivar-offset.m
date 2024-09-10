// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -emit-llvm %s -o - | FileCheck %s

// CHECK: @"OBJC_IVAR_$_StaticLayout.static_layout_ivar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass.superClassIvar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass._superClassProperty" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar" = constant i64 32
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar2" = constant i64 40
// CHECK: @"OBJC_IVAR_$_IntermediateClass._intermediateProperty" = hidden constant i64 48
// CHECK: @"OBJC_IVAR_$_SubClass.subClassIvar" = constant i64 56
// CHECK: @"OBJC_IVAR_$_SubClass._subClassProperty" = hidden constant i64 64
// CHECK: @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar" = hidden global i64 12

@interface NSObject {
  int these, will, never, change, ever;
}
@end

@interface StaticLayout : NSObject
@end

@implementation StaticLayout {
  int static_layout_ivar;
}

// CHECK-LABEL: define internal void @"\01-[StaticLayout meth]"
-(void)meth {
  static_layout_ivar = 0;
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_StaticLayout
  // CHECK: getelementptr inbounds i8, ptr %0, i64 20
}
@end

@interface SuperClass : NSObject
@property (nonatomic, assign) int superClassProperty;
@end

@implementation SuperClass {
  int superClassIvar; // Declare an ivar
}

// CHECK-LABEL: define internal void @"\01-[SuperClass superClassMethod]"
- (void)superClassMethod {
    _superClassProperty = 42;
    superClassIvar = 10;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SuperClass
    // CHECK: getelementptr inbounds i8, ptr %1, i64 20
}

// Implicitly synthesized method here
// CHECK-LABEL: define internal i32 @"\01-[SuperClass superClassProperty]"
// CHECK: getelementptr inbounds i8, ptr %0, i64 24

// CHECK-LABEL: define internal void @"\01-[SuperClass setSuperClassProperty:]"
// CHECK: getelementptr inbounds i8, ptr %1, i64 24
@end

@interface IntermediateClass : SuperClass {
    double intermediateClassIvar;

    @protected
    int intermediateClassIvar2;
}
@property (nonatomic, strong) SuperClass *intermediateProperty;
@end

@implementation IntermediateClass
@synthesize intermediateProperty = _intermediateProperty;

// CHECK-LABEL: define internal void @"\01-[IntermediateClass intermediateClassMethod]"
- (void)intermediateClassMethod {
    intermediateClassIvar = 3.14;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_IntermediateClass
    // CHECK: getelementptr inbounds i8, ptr %0, i64 32
}

// CHECK-LABEL: define internal void @"\01-[IntermediateClass intermediateClassPropertyMethod]"
- (void)intermediateClassPropertyMethod {
    self.intermediateProperty = 0;
    // CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_
    // CHECK: call void @objc_msgSend(ptr noundef %0, ptr noundef %1, ptr noundef null)
}

// CHECK-LABEL: define internal void @"\01-[IntermediateClass intermediateClassPropertyMethodDirect]"
- (void)intermediateClassPropertyMethodDirect {
    _intermediateProperty = 0;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_IntermediateClass._intermediateProperty"
    // CHECK: getelementptr inbounds i8, ptr %0, i64 48
}
@end

@interface SubClass : IntermediateClass {
    double subClassIvar;
}
@property (nonatomic, assign) SubClass *subClassProperty;
@end

@implementation SubClass

// CHECK-LABEL: define internal void @"\01-[SubClass subclassVar]"
- (void)subclassVar {
    subClassIvar = 6.28;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SubClass
    // CHECK: getelementptr inbounds i8, ptr %0, i64 56
}

// CHECK-LABEL: define internal void @"\01-[SubClass intermediateSubclassVar]"
-(void)intermediateSubclassVar {
    intermediateClassIvar = 3.14;
    // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_IntermediateClass
    // CHECK: getelementptr inbounds i8, ptr %0, i64 32
}

// Implicit synthesized method here:
// CHECK-LABEL: define internal ptr @"\01-[SubClass subClassProperty]"
// CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SubClass._subClassProperty"
// CHECK: getelementptr inbounds i8, ptr %0, i64 64

// CHECK-LABEL: define internal void @"\01-[SubClass setSubClassProperty:]"
// CHECK-NOT: load i64, ptr @"OBJC_IVAR_$_SubClass._subClassProperty"
// CHECK: getelementptr inbounds i8, ptr %1, i64 64
@end

@interface NotNSObject {
  int these, might, change;
}
@end

@interface NotStaticLayout : NotNSObject
@end

@implementation NotStaticLayout {
  int not_static_layout_ivar;
}

// CHECK-LABEL: define internal void @"\01-[NotStaticLayout meth]"
-(void)meth {
  not_static_layout_ivar = 0;
  // CHECK: load i64, ptr @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar
}
@end
