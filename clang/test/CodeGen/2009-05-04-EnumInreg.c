// RUN: %clang_cc1 -no-enable-noundef-analysis -emit-llvm -triple i686-apple-darwin -mregparm 3 %s -o - | FileCheck %s
// PR3967

enum kobject_action {
        KOBJ_ADD,
        KOBJ_REMOVE,
        KOBJ_CHANGE,
        KOBJ_MOVE,
        KOBJ_ONLINE,
        KOBJ_OFFLINE,
        KOBJ_MAX
};

struct kobject;

// CHECK: i32 inreg %action
void kobject_uevent(struct kobject *kobj, enum kobject_action action) {}
