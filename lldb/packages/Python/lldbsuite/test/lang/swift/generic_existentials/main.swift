class Pat : Error {
  var x : Int
  init(_ x : Int) {
    self.x = x
  }
}

func f<T>(_ x : T) -> T {
  return x //%self.expect("frame var -d run-target -- x", substrs=['(a.Pat) x', '(x = 23)'])
           //%self.expect("expr -d run-target -- x", substrs=['(a.Pat) $R', '(x = 23)'])
}

let patatino = Pat(23) as AnyObject
f(patatino)
