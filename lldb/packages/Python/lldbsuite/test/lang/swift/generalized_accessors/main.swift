extension Array where Element == Int {
  mutating func double() {
    for i in indices {
      self[i] *= 2 // break here
    }
  }
}

var d: [String:[Int]]  = [
  "odd":[1,3,5,7,9],
  "even":[222,444,666,888]
]

d["odd"]?.double()
