// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-FRAGILE < %t %s

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
// RUN: FileCheck --check-prefix CHECK-NONFRAGILE < %t %s

// CHECK-FRAGILE:      !llvm.module.flags = !{!0, !1, !2, !3}
// CHECK-FRAGILE:      !0 = metadata !{i32 1, metadata !"Objective-C Version", i32 1}
// CHECK-FRAGILE-NEXT: !1 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
// CHECK-FRAGILE-NEXT: !2 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__OBJC, __image_info,regular"}
// CHECK-FRAGILE-NEXT: !3 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}

// CHECK-NONFRAGILE:      !llvm.module.flags = !{!0, !1, !2, !3}
// CHECK-NONFRAGILE:      !0 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
// CHECK-NONFRAGILE-NEXT: !1 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
// CHECK-NONFRAGILE-NEXT: !2 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
// CHECK-NONFRAGILE-NEXT: !3 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
