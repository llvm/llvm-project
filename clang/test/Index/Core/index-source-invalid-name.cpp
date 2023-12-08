// XFAIL: target={{.*}}-aix{{.*}}
// RUN: c-index-test core -print-source-symbols -- %s -std=c++1z -target x86_64-apple-macosx10.7 | FileCheck %s

namespace rdar32474406 {
// CHECK: [[@LINE+1]]:6 | function/C | foo | c:@N@rdar32474406@F@foo# | __ZN12rdar324744063fooEv | Decl,RelChild | rel: 1
void foo();
// CHECK: [[@LINE+1]]:16 | type-alias/C | Func_t | c:index-source-invalid-name.cpp@N@rdar32474406@T@Func_t | <no-cgname> | Def,RelChild | rel: 1
typedef void (*Func_t)();
// CHECK: [[@LINE+4]]:1 | type-alias/C | Func_t | c:index-source-invalid-name.cpp@N@rdar32474406@T@Func_t | <no-cgname> | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | rdar32474406 | c:@N@rdar32474406
// CHECK: [[@LINE+2]]:14 | function/C | foo | c:@N@rdar32474406@F@foo# | __ZN12rdar324744063fooEv | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | rdar32474406 | c:@N@rdar32474406
Func_t[] = { foo }; // invalid decomposition
}
