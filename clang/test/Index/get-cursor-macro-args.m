// Test without PCH
// RUN: c-index-test -cursor-at=%S/get-cursor-macro-args.h:9:12 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:21 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:9 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:22 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:15:12 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:15:20 \
// RUN:       %s -include %S/get-cursor-macro-args.h | FileCheck %s

// Test with PCH
// RUN: c-index-test -write-pch %t.pch -x objective-c-header %S/get-cursor-macro-args.h
// RUN: c-index-test -cursor-at=%S/get-cursor-macro-args.h:9:12 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:21 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:9 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:9:22 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:15:12 \
// RUN:              -cursor-at=%S/get-cursor-macro-args.h:15:20 \
// RUN:       %s -include-pch %t.pch | FileCheck %s

// CHECK:      ObjCClassRef=MyClass:1:12
// CHECK-NEXT: ObjCMessageExpr=meth:2:8
// CHECK-NEXT: ObjCMessageExpr=meth:2:8
// CHECK-NEXT: ObjCMessageExpr=meth:2:8
// CHECK-NEXT: ObjCMessageExpr=meth:2:8
// CHECK-NEXT: ObjCClassRef=MyClass:1:12
