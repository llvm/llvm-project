// RUN: %clang -target x86_64-unkown-unknown -fms-extensions -rewrite-objc %s -o - | FileCheck %s
// rdar://12189793

#ifdef __cplusplus

void *sel_registerName(const char *);

@interface Root @end

@interface MYINTF : Root
@end

#endif

@implementation MYINTF 
- (id) MYMETH { return [self MYMETH]; }
@end

int main() {
}

// CHECK: static struct _class_ro_t _OBJC_CLASS_RO_$_MYINTF
// CHECK-NEXT: 0, 0, 0,
// CHECK-NEXT: (unsigned int)0,
// CHECK-NEXT: 0,
// CHECK-NEXT: "MYINTF",
