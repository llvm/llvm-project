// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-objc-root-class -emit-llvm -o - %s | FileCheck %s

// -- classref for the message send in main()
//
// The class is declared with objc_class_stub, so LSB of the class pointer
// must be set to 1.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr getelementptr (i8, ptr @"OBJC_CLASS_$_Base", i32 1), align 8

// -- classref for the super message send in anotherClassMethod()
//
// Metaclasses do not use the "stub" mechanism and are referenced statically.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_SUP_REFS_$_" = private global ptr @"OBJC_METACLASS_$_Derived", section "__DATA,__objc_superrefs,regular,no_dead_strip", align 8

// -- classref for the super message send in anotherInstanceMethod()
//
// The class is declared with objc_class_stub, so LSB of the class pointer
// must be set to 1.
//
// CHECK-LABEL: @"OBJC_CLASSLIST_SUP_REFS_$_.1" = private global ptr getelementptr (i8, ptr @"OBJC_CLASS_$_Derived", i32 1), section "__DATA,__objc_superrefs,regular,no_dead_strip", align 8

// -- category list for class stubs goes in __objc_catlist2.
//
// CHECK-LABEL: @"OBJC_LABEL_STUB_CATEGORY_$" = private global [1 x ptr] [ptr @"_OBJC_$_CATEGORY_Derived_$_MyCategory"], section "__DATA,__objc_catlist2,regular,no_dead_strip", align 8

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

int main(void) {
  [Base classMethod];
}
// CHECK-LABEL: define{{.*}} i32 @main()
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[CLASS:%.*]] = call ptr @objc_loadClassref(ptr @"OBJC_CLASSLIST_REFERENCES_$_")
// CHECK-NEXT:   [[SELECTOR:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT:   call void @objc_msgSend(ptr noundef [[CLASS]], ptr noundef [[SELECTOR]])
// CHECK-NEXT:   ret i32 0

// CHECK-LABEL: declare extern_weak ptr @objc_loadClassref(ptr)
// CHECK-SAME: [[ATTRLIST:#.*]]

@implementation Derived (MyCategory)

+ (void) anotherClassMethod {
  [super classMethod];
}
// CHECK-LABEL: define internal void @"\01+[Derived(MyCategory) anotherClassMethod]"(ptr noundef %self, ptr noundef %_cmd) #0 {
// CHECK-NEXT: entry:
// CHECK:        [[SUPER:%.*]] = alloca %struct._objc_super, align 8
// CHECK:        [[METACLASS_REF:%.*]] = load ptr, ptr @"OBJC_CLASSLIST_SUP_REFS_$_", align 8
// CHECK:        [[DEST:%.*]] = getelementptr inbounds nuw %struct._objc_super, ptr [[SUPER]], i32 0, i32 1
// CHECK:        store ptr [[METACLASS_REF]], ptr [[DEST]], align 8
// CHECK:        call void @objc_msgSendSuper2(ptr noundef [[SUPER]], ptr noundef {{%.*}})
// CHECK:        ret void

- (void) anotherInstanceMethod {
  [super instanceMethod];
}
// CHECK-LABEL: define internal void @"\01-[Derived(MyCategory) anotherInstanceMethod]"(ptr noundef %self, ptr noundef %_cmd) #0 {
// CHECK-NEXT: entry:
// CHECK:        [[SUPER:%.*]] = alloca %struct._objc_super, align 8
// CHECK:        [[CLASS_REF:%.*]] = call ptr @objc_loadClassref(ptr @"OBJC_CLASSLIST_SUP_REFS_$_.1")
// CHECK:        [[DEST:%.*]] = getelementptr inbounds nuw %struct._objc_super, ptr [[SUPER]], i32 0, i32 1
// CHECK:        store ptr [[CLASS_REF]], ptr [[DEST]], align 8
// CHECK:        call void @objc_msgSendSuper2(ptr noundef [[SUPER]], ptr noundef {{%.*}})
// CHECK:        ret void

@end

// -- calls to objc_loadClassRef() are readnone
// CHECK: attributes [[ATTRLIST]] = { nounwind nonlazybind memory(none) }
