import ObjCModule

func use<T>(_ t: T) {}

let pureSwift = 42
let obj = ObjCClass()
let proto = getProto()
use(pureSwift) // break here
use(obj)
use(proto)
