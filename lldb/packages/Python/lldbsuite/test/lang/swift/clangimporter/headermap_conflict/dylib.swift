import Foo

func use<T>(_ t: T) {}
// In a struct to thwart DWARFImporter.
// Generic to force scratch context.
struct S<T> {
    let elem = FooNested(i: 23)
}

public func f<T>(_ t: T) {
  let foo = S<T>()
  use(foo) // break here
}
