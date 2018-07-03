func use<T>(_ t: T) {}

public struct Foo {
  let i = 23
  public init() {}
  public func f() {
    use(i)
  }
}
