import CBar
import Foundation

func use<T>(_ t: T) {}

@objc public class Bar : NSObject {
  @objc public func f() {
    let bar = CBar(j: 42)
    use(bar) // break here
  }
}
