import SomeLibrary

func stop() {}

func test() {
  let container = ContainsTwoInts(10)
  let simple = 1
  stop() // break here
}

test()
