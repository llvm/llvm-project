// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -emit-llvm %s -o - | FileCheck %s

// CHECK: @"OBJC_IVAR_$_StaticLayout.static_layout_ivar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass.superClassIvar" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_SuperClass._superClassProperty" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar" = constant i64 32
// CHECK: @"OBJC_IVAR_$_IntermediateClass.intermediateClassIvar2" = constant i64 40
// CHECK: @"OBJC_IVAR_$_IntermediateClass._intermediateProperty" = hidden constant i64 48
// CHECK: @"OBJC_IVAR_$_SubClass.subClassIvar" = constant i64 56
// CHECK: @"OBJC_IVAR_$_SubClass._subClassProperty" = hidden constant i64 64

// CHECK: @"OBJC_IVAR_$_RootClass.these" = constant i64 0
// CHECK: @"OBJC_IVAR_$_RootClass.never" = constant i64 4
// CHECK: @"OBJC_IVAR_$_RootClass.change" = constant i64 8
// CHECK: @"OBJC_IVAR_$_StillStaticLayout.static_layout_ivar" = hidden constant i64 12

// CHECK: @"OBJC_IVAR_$_NotStaticLayout.not_static_layout_ivar" = hidden global i64 12
// CHECK: @"OBJC_IVAR_$_SuperClass2._superClassProperty2" = hidden constant i64 20
// CHECK: @"OBJC_IVAR_$_IntermediateClass2._IntermediateClass2Property" = hidden constant i64 24
// CHECK: @"OBJC_IVAR_$_SubClass2._subClass2Property" = hidden constant i64 28

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

 __attribute((objc_root_class))  @interface RootClass {
  int these, never, change;
}
@end

@implementation RootClass 
@end

@interface StillStaticLayout : RootClass
@end

@implementation StillStaticLayout {
  int static_layout_ivar;
}

// CHECK-LABEL: define internal void @"\01-[StillStaticLayout meth]"
-(void)meth {
  static_layout_ivar = 0;
  // CHECK-NOT: load i64, ptr @"OBJC_IVAR_$StillStaticLayout.static_layout_ivar
}
@end

@interface NotNSObject
@end

@interface NotStaticLayout : NotNSObject {
  int these, might, change;
}
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

// CHECK: define internal i32 @"\01-[IntermediateClass2 IntermediateClass2Property]"(ptr noundef %[[SELF:.*]],
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 24
// CHECK: %[[LOAD:.*]] = load atomic i32, ptr %[[ADD_PTR]] unordered, align 4
// CHECK: ret i32 %[[LOAD]]

// CHECK: define internal i32 @"\01-[SubClass2 subClass2Property]"(ptr noundef %[[SELF:.*]],
// CHECK: %[[SELF_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[SELF]], ptr %[[SELF_ADDR]], align 8
// CHECK: %[[V0:.*]] = load ptr, ptr %[[SELF_ADDR]], align 8
// CHECK: %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[V0]], i64 28
// CHECK: %[[LOAD:.*]] = load atomic i32, ptr %[[ADD_PTR]] unordered, align 4
// CHECK: ret i32 %[[LOAD]]

@interface SuperClass2 : NSObject
@property int superClassProperty2;
@end

@interface IntermediateClass2 : SuperClass2
@property int IntermediateClass2Property;
@end

@interface IntermediateClass3 : SuperClass2
@property int IntermediateClass3Property;
@end

@interface SubClass2 : IntermediateClass2
@property int subClass2Property;
@end

@implementation IntermediateClass3
@end

@implementation SuperClass2
@end

@implementation IntermediateClass2
@end

@implementation SubClass2
@end
