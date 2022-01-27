// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -emit-llvm -o - %s | FileCheck %s

// -- classref for the message send in main()
//
// The class is declared with objc_class_stub, so LSB of the class pointer
// must be set to 1.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_REFERENCES_$_" = internal global i8* getelementptr (i8, i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Base" to i8*), i32 1), align 8

// -- classref for the super message send in anotherClassMethod()
//
// Metaclasses do not use the "stub" mechanism and are referenced statically.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_SUP_REFS_$_" = private global %struct._class_t* @"OBJC_METACLASS_$_Derived", section "__DATA,__objc_superrefs,regular,no_dead_strip", align 8

// -- classref for the super message send in anotherInstanceMethod()
//
// The class is declared with objc_class_stub, so LSB of the class pointer
// must be set to 1.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_SUP_REFS_$_.1" = private global i8* getelementptr (i8, i8* bitcast (%struct._class_t* @"OBJC_CLASS_$_Derived" to i8*), i32 1), section "__DATA,__objc_superrefs,regular,no_dead_strip", align 8

// -- category list for class stubs goes in __objc_catlist2.
//
// CHECK-LABEL: @"OBJC_LABEL_STUB_CATEGORY_$" = private global [1 x i8*] [i8* bitcast (%struct._category_t* @"_OBJC_$_CATEGORY_Derived_$_MyCategory" to i8*)], section "__DATA,__objc_catlist2,regular,no_dead_strip", align 8

__attribute__((objc_class_stub))
__attribute__((objc_subclassing_restricted))
@interface Base
+ (void) classMethod;
- (void) instanceMethod;
@end

__attribute__((objc_class_stub))
__attribute__((objc_subclassing_restricted))
@interface Derived : Base
@end

int main() {
  [Base classMethod];
}
// CHECK-LABEL: define{{.*}} i32 @main()
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[CLASS:%.*]] = call %struct._class_t* @objc_loadClassref(i8** @"OBJC_CLASSLIST_REFERENCES_$_")
// CHECK-NEXT:   [[RECEIVER:%.*]] = bitcast %struct._class_t* [[CLASS]] to i8*
// CHECK-NEXT:   [[SELECTOR:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT:   call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*)*)(i8* noundef [[RECEIVER]], i8* noundef [[SELECTOR]])
// CHECK-NEXT:   ret i32 0

// CHECK-LABEL: declare extern_weak %struct._class_t* @objc_loadClassref(i8**)
// CHECK-SAME: [[ATTRLIST:#.*]]

@implementation Derived (MyCategory)

+ (void) anotherClassMethod {
  [super classMethod];
}
// CHECK-LABEL: define internal void @"\01+[Derived(MyCategory) anotherClassMethod]"(i8* noundef %self, i8* noundef %_cmd) #0 {
// CHECK-NEXT: entry:
// CHECK:        [[SUPER:%.*]] = alloca %struct._objc_super, align 8
// CHECK:        [[METACLASS_REF:%.*]] = load %struct._class_t*, %struct._class_t** @"OBJC_CLASSLIST_SUP_REFS_$_", align 8
// CHECK:        [[CAST_METACLASS_REF:%.*]] = bitcast %struct._class_t* [[METACLASS_REF]] to i8*
// CHECK:        [[DEST:%.*]] = getelementptr inbounds %struct._objc_super, %struct._objc_super* [[SUPER]], i32 0, i32 1
// CHECK:        store i8* [[CAST_METACLASS_REF]], i8** [[DEST]], align 8
// CHECK:        call void bitcast (i8* (%struct._objc_super*, i8*, ...)* @objc_msgSendSuper2 to void (%struct._objc_super*, i8*)*)(%struct._objc_super* noundef [[SUPER]], i8* noundef {{%.*}})
// CHECK:        ret void

- (void) anotherInstanceMethod {
  [super instanceMethod];
}
// CHECK-LABEL: define internal void @"\01-[Derived(MyCategory) anotherInstanceMethod]"(%0* noundef %self, i8* noundef %_cmd) #0 {
// CHECK-NEXT: entry:
// CHECK:        [[SUPER:%.*]] = alloca %struct._objc_super, align 8
// CHECK:        [[CLASS_REF:%.*]] = call %struct._class_t* @objc_loadClassref(i8** @"OBJC_CLASSLIST_SUP_REFS_$_.1")
// CHECK:        [[CAST_CLASS_REF:%.*]] = bitcast %struct._class_t* [[CLASS_REF]] to i8*
// CHECK:        [[DEST:%.*]] = getelementptr inbounds %struct._objc_super, %struct._objc_super* [[SUPER]], i32 0, i32 1
// CHECK:        store i8* [[CAST_CLASS_REF]], i8** [[DEST]], align 8
// CHECK:        call void bitcast (i8* (%struct._objc_super*, i8*, ...)* @objc_msgSendSuper2 to void (%struct._objc_super*, i8*)*)(%struct._objc_super* noundef [[SUPER]], i8* noundef {{%.*}})
// CHECK:        ret void

@end

// -- calls to objc_loadClassRef() are readnone
// CHECK: attributes [[ATTRLIST]] = { nounwind nonlazybind readnone }
