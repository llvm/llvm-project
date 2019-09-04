import CFoo
import Foundation

func use<T>(_ t: T) {}

@objc public class Foo : NSObject {
  @objc public func f() {
    let foo = CFoo(i: 23)
    use(foo) // break here
  }
}
