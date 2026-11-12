// RUN: c-index-test -print-usr-file %s | FileCheck %s
// This isn't really C code; it has a .c extension to get picked up by lit.
ObjCClass NSObject
ObjCCategory NSObject foo
ObjCIvar x c:objc(cs)NSObject
ObjCMethod foo: 0 c:objc(cs)NSObject
ObjCMethod baz:with 1 c:objc(cs)NSObject
ObjCProperty gimme c:objc(cs)NSObject
ObjCProtocol blah
// CHECK: c:objc(cs)NSObject
// CHECK: c:objc(cy)NSObject@foo
// CHECK: c:objc(cs)NSObject@x
// CHECK: c:objc(cs)NSObject(cm)foo:
// CHECK: c:objc(cs)NSObject(im)baz:with
// CHECK: c:objc(cs)NSObject(py)gimme
// CHECK: c:objc(pl)blah

