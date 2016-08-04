import Foundation
///
Data()
///
enum VagueProblem: Error { case SomethingWentWrong }
///
func foo() throws -> Int { throw VagueProblem.SomethingWentWrong }
///
foo()
///\$E0
///SomethingWentWrong
