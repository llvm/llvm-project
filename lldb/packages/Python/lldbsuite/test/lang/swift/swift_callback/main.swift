func takeCallback(body: (_ line: String, _ lineNum: Int, _ stop: inout Bool) -> Void) -> Void {
  var stop: Bool = false
  body("Hello", 3, &stop) //%self.expect('frame variable -d run -- stop', substrs=['(Bool) stop = false'])
                          //%self.expect('expr -d run -- stop', substrs=['(Bool) $R0 = false'])
                          //%self.expect('frame variable -d run -- body', substrs=['closure #1 (Swift.String, Swift.Int, inout Swift.Bool) -> ()'])
}
let b = { (line: String, lineNum: Int, stop: inout Bool) -> Void in }

takeCallback(body: b)
