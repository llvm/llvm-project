// Test without PCH
// RUN: c-index-test -test-annotate-tokens=%S/annotate-macro-args.h:9:1:10:1 %s -include %S/annotate-macro-args.h | FileCheck -check-prefix=CHECK1 %s
// RUN: c-index-test -test-annotate-tokens=%S/annotate-macro-args.h:15:1:16:1 %s -include %S/annotate-macro-args.h | FileCheck -check-prefix=CHECK2 %s

// Test with PCH
// RUN: c-index-test -write-pch %t.pch -x objective-c-header %S/annotate-macro-args.h -Xclang -detailed-preprocessing-record
// RUN: c-index-test -test-annotate-tokens=%S/annotate-macro-args.h:9:1:10:1 %s -include-pch %t.pch | FileCheck -check-prefix=CHECK1 %s
// RUN: c-index-test -test-annotate-tokens=%S/annotate-macro-args.h:15:1:16:1 %s -include-pch %t.pch | FileCheck -check-prefix=CHECK2 %s

// CHECK1: Identifier: "MACRO" [9:3 - 9:8] macro expansion=MACRO:6:9
// CHECK1: Punctuation: "(" [9:8 - 9:9]
// CHECK1: Punctuation: "[" [9:9 - 9:10] ObjCMessageExpr=meth:2:8
// CHECK1: Identifier: "MyClass" [9:10 - 9:17] ObjCClassRef=MyClass:1:12
// CHECK1: Identifier: "meth" [9:18 - 9:22] ObjCMessageExpr=meth:2:8
// CHECK1: Punctuation: "]" [9:22 - 9:23] ObjCMessageExpr=meth:2:8
// CHECK1: Punctuation: ")" [9:23 - 9:24]

// CHECK2: Identifier: "INVOKE" [15:3 - 15:9] macro expansion=INVOKE:12:9
// CHECK2: Punctuation: "(" [15:9 - 15:10]
// CHECK2: Identifier: "meth" [15:10 - 15:14] ObjCMessageExpr=meth:2:8
// CHECK2: Punctuation: "," [15:14 - 15:15]
// CHECK2: Identifier: "MyClass" [15:16 - 15:23] ObjCClassRef=MyClass:1:12
// CHECK2: Punctuation: ")" [15:23 - 15:24]
