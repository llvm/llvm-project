import Foundation

public protocol LibraryProtocol : class {}

public final class Foo : NSObject {
  public init(_ input : LibraryProtocol) {
    // When evaluating "input" here, RemoteAST will try to get its
    // dynamic type.  This must *not* trigger an import of the "main"
    // module in the Library module context.
    print("break here")
  }
}
