// Let's try to make sure that frame var prints the dynamic type of a class
// (and a subclass) conforming to Error correctly (and that we don't crash).
// This involves resolving the dynamic type correctly in the language runtime.

class MyErr : Error {
  var x : Int

  init(_ x : Int) {
    self.x = x
  }
}

class MyOtherErr : MyErr {}

func f<T>(_ Pat : T) -> T {
  return Pat //%self.expect('frame variable -d run -- Pat', substrs=['MyErr', 'x = 23'])
}

func g<T>(_ Pat : T) -> T {
  return Pat //%self.expect('frame variable -d run -- Pat', substrs=['MyOtherErr', 'x = 42'])
}

func main() -> Int {
  let foo : MyErr = MyErr(23)
  let goo : MyErr = MyOtherErr(42)
  let patatino = f(foo)
  let tinky = g(goo)
  return 0
}

let _ = main()
