class PayloadErr : Error {
  var x : Int

  init(_ x : Int) {
    self.x = x
  }
}

class MyOtherErr : PayloadErr {}

enum CErr : Error {
  case Topolino
  case Paperino
}

func g<T, U>(_ tuple : (T, U)) -> T {
  return tuple.0 //%self.expect("frame var -d run-target -- tuple",
                 //%            substrs=['(a.CErr, Int)', 'Topolino', '42'])
}

func h<U, V>(_ tuple : (U, V)) -> (U, V) {
  return tuple //%self.expect("frame var -d run-target -- tuple", 
               //%             substrs=['(a.PayloadErr, a.MyOtherErr) tuple',
               //%                      '0 =', '(x = 23)',
               //%                      'a.PayloadErr = {', 'x = 42'])
               //%self.expect("expr -d run-target -- tuple",
               //%             substrs=['(a.PayloadErr, a.PayloadErr) $R',
               //%                      '0 =', '(x = 23)',
               //%                      'a.PayloadErr = {', 'x = 42'])
}

g((CErr.Topolino as Error, 42))
h((PayloadErr(23), MyOtherErr(42) as PayloadErr))
