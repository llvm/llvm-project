import Darwin
import Foundation
import ObjCStuff

func use<T>(_ t: T) {}

func f() {
  // We never imported Darwin, so this is all we get.
  // CHECK-DAG: (size_t) ctype = 1024
  let ctype = size_t(1024)
  // This works as a Clang type via the Objective-C runtime.
  // CHECK-DAG: (ObjCClass) object = 0x{{[0-9a-f]+$}}
  // CHECK-DAG: (ObjCClass) object = {{.*}}Hello from Objective-C!
  let object = ObjCClass()
  // The Objective-C runtime recognizes this as a tagged pointer.
  // CHECK-DAG: (__NSCFNumber) inlined = {{.*}}42
  let inlined = NSNumber(value: 42)
  // CHECK-DAG: (CMYK) enumerator = yellow
  let enumerator = yellow
  // CHECK-DAG: (FourColors) typedef = cyan
  let typedef = FourColors(0)
  let union = Union(i: 23)
  use(ctype) // break here
  use(object)
  use(inlined)
  use(enumerator)
  use(typedef)
  use(union)
}

f()
