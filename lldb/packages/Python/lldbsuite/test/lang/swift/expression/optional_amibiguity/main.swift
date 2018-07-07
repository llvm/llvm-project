import Foundation
import Optional

class MyObject : NSObject
{
  func callClosure(_ closure : () -> Void) {
    closure()
  }

  func doCall() {
    callClosure() { [weak self] in
      if let real_self = self {
        print(real_self) // Set a breakpoint here
      }
    }
  }
}

var my_object = MyObject()
my_object.doCall()
