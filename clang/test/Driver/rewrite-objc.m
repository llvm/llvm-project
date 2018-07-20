// RUN: %clang -no-canonical-prefixes -target x86_64-apple-macosx10.7.0 -rewrite-objc %s -o - -### 2>&1 | \
// RUN:   FileCheck -check-prefix=TEST0 %s
// TEST0: clang{{.*}}" "-cc1"
// TEST0: "-rewrite-objc"
// FIXME: CHECK-NOT is broken somehow, it doesn't work here. Check adjacency instead.
// TEST0: "-fmessage-length" "0" "-stack-protector" "1" "-fblocks" "-fencode-extended-block-signature" "-fregister-global-dtors-with-atexit" "-fobjc-runtime=macosx" "-fno-objc-infer-related-result-type" "-fobjc-exceptions" "-fexceptions" "-fmax-type-align=16" "-fdiagnostics-show-option"
