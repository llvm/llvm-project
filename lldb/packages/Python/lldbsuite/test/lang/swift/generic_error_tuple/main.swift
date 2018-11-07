class MyErr : Error {
  var x : Int

  init(_ x : Int) {
    self.x = x
  }
}

class MyOtherErr : MyErr {}

func h<U, V>(_ tuple : (U, V)) -> (U, V) {
  return tuple //%self.expect("frame var -d run-target -- tuple", 
               //%             substrs=['(a.MyErr, a.MyErr) tuple',
               //%                      '0 =', '(x = 23)',
               //%                      'a.MyErr = {', 'x = 42'])
               //%self.expect("expr -d run-target -- tuple",
               //%             substrs=['(a.MyErr, a.MyErr) $R',
               //%                      '0 =', '(x = 23)',
               //%                      'a.MyErr = {', 'x = 42'])
}

func main() -> Int {
  let foo : MyErr = MyErr(23)
  let goo : MyErr = MyOtherErr(42)
  h((foo, goo))
  return 0
}

let _ = main()
