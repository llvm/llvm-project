@_implementationOnly import SomeLibraryCore

internal class BoxedTwoInts {
  internal var value: TwoInts
  init(_ value: TwoInts) { self.value = value }
}

public struct ContainsTwoInts {
  internal var wrapped: BoxedTwoInts
  public var other: Int

  public init(_ value: Int) {
    wrapped = BoxedTwoInts(.init(2, 3))
    other = value
  }
}
