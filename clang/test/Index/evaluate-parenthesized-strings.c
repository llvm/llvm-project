const char *plain = "hello";
const char *paren = ("hello");
const char *nested = ((("hello")));

// RUN: c-index-test -evaluate-cursor-at=%s:1:13 \
// RUN:   -evaluate-cursor-at=%s:2:13 \
// RUN:   -evaluate-cursor-at=%s:3:13 %s | FileCheck %s
// CHECK: VarDecl=plain:{{.*}}Kind: CString , Value: hello
// CHECK: VarDecl=paren:{{.*}}Kind: CString , Value: hello
// CHECK: VarDecl=nested:{{.*}}Kind: CString , Value: hello
