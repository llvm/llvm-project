import Foundation

func f() {
  let object = NSNumber(value: 42) // FIXME: CHECK: objexct = <could not resolve type>
  print(object) // break here
}

f()
