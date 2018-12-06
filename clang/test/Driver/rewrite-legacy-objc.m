// RUN: %clang -no-canonical-prefixes -target x86_64-apple-macosx10.7.0 -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: clang{{.*}}" "-cc1"
// TEST0: "-rewrite-objc"
// FIXME: CHECK-NOT is broken somehow, it doesn't work here. Check adjacency instead.
// TEST0: "-fmessage-length" "0" "-stack-protector" "1" "-fblocks" "-fencode-extended-block-signature" "-fregister-global-dtors-with-atexit" "-fobjc-runtime=macosx-fragile" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fexceptions" "-fmax-type-align=16" "-fdiagnostics-show-option"
// TEST0: rewrite-legacy-objc.m"
// RUN: %clang -no-canonical-prefixes -target i386-apple-macosx10.9.0 -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST1 %s
// RUN: %clang -no-canonical-prefixes -target i386-apple-macosx10.6.0 -rewrite-legacy-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST2 %s
// TEST1: "-fmessage-length" "0" "-stack-protector" "1" "-fblocks" "-fencode-extended-block-signature" "-fregister-global-dtors-with-atexit" "-fobjc-runtime=macosx-fragile" "-fobjc-subscripting-legacy-runtime" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fmax-type-align=16" "-fdiagnostics-show-option"
// TEST2: "-fmessage-length" "0" "-stack-protector" "1" "-fblocks" "-fencode-extended-block-signature" "-fregister-global-dtors-with-atexit" "-fobjc-runtime=macosx-fragile" "-fobjc-subscripting-legacy-runtime" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fmax-type-align=16" "-fdiagnostics-show-option"
