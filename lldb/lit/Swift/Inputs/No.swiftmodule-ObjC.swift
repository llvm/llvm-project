import Darwin
import Foundation
import ObjCStuff

func f() {
  // We never imported Darwin, so this is all we get.
  // CHECK-DAG: (Builtin.RawPointer) ctype = 0x0000000000000400
  let ctype = size_t(1024)
  // This works as a Clang type via the Objective-C runtime.
  // CHECK-DAG: (ObjCClass) object = 0x{{[0-9a-f]+$}}
  // CHECK-DAG: (ObjCClass) object = {{.*}}Hello from Objective-C!
  let object = ObjCClass()
  // The Objective-C runtime recognizes this as a tagged pointer.
  // CHECK-DAG: (__NSCFNumber) inlined = {{.*}}42
  let inlined = NSNumber(value: 42)
  print(object) // break here
}

f()
