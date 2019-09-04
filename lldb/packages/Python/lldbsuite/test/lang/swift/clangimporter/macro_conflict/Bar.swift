import WithMacro
import Foundation

func use<T>(_ t: T) {}

@objc public class Bar : NSObject {
  @objc public func f() {
    let bar = CBar(i: 23)
    use(bar) // break here
  }
}
