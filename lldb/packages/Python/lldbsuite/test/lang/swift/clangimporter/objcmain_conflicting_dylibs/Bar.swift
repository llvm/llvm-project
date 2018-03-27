import CBar
import Foundation

func use<T>(_ t: T) {}

@objc public class Bar : NSObject {
  @objc public func f() {
    let baz = Baz(i_am_from_Bar: 42)
    use(baz) // break here
  }
}
