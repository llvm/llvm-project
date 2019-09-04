class Optional {
  var patatino : Int
  init(_ patatino : Int) {
    self.patatino = patatino
  }
}

class MyObject
{
  func callClosure(_ closure : () -> Void) {
    closure()
  }

  func doCall() {
    callClosure() { [weak self] in
      if let real_self = self {
        let x : Optional = Optional(25)
        print(real_self) // Set a breakpoint here
      }
    }
  }
}

var my_object = MyObject()
my_object.doCall()
