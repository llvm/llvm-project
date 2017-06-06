// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

// CHECK-FRAGILE:      !llvm.module.flags = !{{{.*}}}
// CHECK-FRAGILE:      !{{[0-9]+}} = !{i32 1, !"Objective-C Version", i32 1}
// CHECK-FRAGILE-NEXT: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Version", i32 0}
// CHECK-FRAGILE-NEXT: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__OBJC,__image_info,regular"}
// CHECK-FRAGILE-NEXT: !{{[0-9]+}} = !{i32 4, !"Objective-C Garbage Collection", i32 0}

// CHECK-NONFRAGILE:      !llvm.module.flags = !{{{.*}}}
// CHECK-NONFRAGILE:      !{{[0-9]+}} = !{i32 1, !"Objective-C Version", i32 2}
// CHECK-NONFRAGILE-NEXT: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Version", i32 0}
// CHECK-NONFRAGILE-NEXT: !{{[0-9]+}} = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
// CHECK-NONFRAGILE-NEXT: !{{[0-9]+}} = !{i32 4, !"Objective-C Garbage Collection", i32 0}
