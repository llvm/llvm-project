import Foundation

func use<T>(_ t: T) {}

@objc public class Foo : NSObject {
  @objc public func f() {
    let foo = 42
    use(foo) // break here
  }
}
