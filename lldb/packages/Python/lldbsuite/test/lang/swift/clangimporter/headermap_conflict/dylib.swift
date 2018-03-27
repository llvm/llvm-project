import Foo

func use<T>(_ t: T) {}

public func f() {
  let foo = FooNested(i: 23)
  use(foo) // break here
}
