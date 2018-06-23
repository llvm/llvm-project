import Darwin
import Foundation

func f() {
  // This works only through a weird side-effect with the NSURL below.
  // CHECK-DAG: (size_t) ctype = 1024
  let ctype = size_t(1024)
  // This works as a Clang type via the Objective-C runtime.
  // CHECK-DAG: (URL) object = "file:///dev/null"
  let object = URL(fileURLWithPath: "/dev/null")
  // The Objective-C runtime recognizes this as a tagged pointer.
  // CHECK-DAG: (__NSCFNumber) inlined = 0x0000000000002a37 42
  let inlined = NSNumber(value: 42)
  print(object) // break here
}

f()
