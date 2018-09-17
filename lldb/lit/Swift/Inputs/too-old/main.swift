struct Point {
  let x: Int = 0
  let y: Int = 0
}

func main() {
  let p = Point()
  print(p) // break here
}

main()

// CHECK: Error while loading Swift module:
// CHECK-NEXT: main: error: module file was created by an older version of the compiler; rebuild '{{.*}}main' and try again:
// CHECK-EMPTY:
// CHECK-NEXT: Debug info from this module will be unavailable in the debugger.
// CHECK-EMPTY:
// CHECK-NEXT: Shared Swift state for main could not be initialized.
// CHECK-NEXT: The REPL and expressions are unavailable.
// CHECK-NOT: error

// Note: This file was compiled with Apple Swift version 2.2 (swiftlang-703.0.18.8 clang-703.0.31).
//       swiftc -g main.swift -o main
