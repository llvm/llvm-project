// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep 'objc_assign' %t | count 0
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep 'objc_assign' %t | count 0

typedef struct {
    int ival;
    id submenu;
} XCBinderContextMenuItem;

id actionMenuForDataNode(void) {
    XCBinderContextMenuItem menusToCreate[]  = {
        {1, 0}
    };
    return 0;
}

XCBinderContextMenuItem GmenusToCreate[]  = {
        {1, 0}
};
