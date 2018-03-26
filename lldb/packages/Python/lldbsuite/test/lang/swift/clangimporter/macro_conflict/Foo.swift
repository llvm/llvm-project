import WithMacro
import Foundation

func use<T>(_ t: T) {}

@objc public class Foo : NSObject {
  @objc public func f() {
    let foo = CFoo(i: 42)
    use(foo) // break here
  }
}
