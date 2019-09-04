class MyClass {
  var x: Int
  init(_ x: Int) {
    self.x = x
  }
}

func f<T>(_ x : T) -> T {
  return x //%self.expect("frame var -d run-target -- x", substrs=['(a.MyClass) x', '(x = 23)'])
           //%self.expect("expr -d run-target -- x", substrs=['(a.MyClass) $R', '(x = 23)'])
}

f(MyClass(23) as Any)
f(MyClass(23) as AnyObject)

func g<T>(_ x : T) -> T {
  return x //%self.expect("frame var -d run-target -- x", substrs=['(a.MyStruct) x', '(x = 23)'])
           //%self.expect("expr -d run-target -- x", substrs=['(a.MyStruct) $R', '(x = 23)'])
}

struct MyStruct {
  var x: Int

  init(_ x: Int) {
    self.x = x
  }
}

g(MyStruct(23) as Any)

func h<T>(_ x : T) -> T {
  return x //%self.expect("frame var -d run-target -- x", substrs=['(a.MyBigStruct) x', '(x = 23, y = 24, z = 25, w = 26)'])
           //%self.expect("expr -d run-target -- x", substrs=['(a.MyBigStruct) $R', '(x = 23, y = 24, z = 25, w = 26)'])
}

struct MyBigStruct {
  var x: Int
  var y: Int
  var z: Int
  var w: Int

  init(_ x: Int) {
    self.x = x
    self.y = x + 1
    self.z = x + 2
    self.w = x + 3
  }
}

h(MyBigStruct(23) as Any)
