func main() -> Int {
  let names = ["foo", "patatino"]

  var reversedNames = names.sorted(by: {
    $0 > $1 } //%self.expect('p $0', substrs=['patatino'])
              //%self.expect('p $1', substrs=['foo'])
              //%self.expect('frame var $0', substrs=['patatino'])
              //%self.expect('frame var $1', substrs=['foo'])
  )

  return 0
}

_ = main()
