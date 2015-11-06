// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
struct Str1 {
  func foo() {}
  func bar() {}
  var a = 3
  var b = "hi"
}

class Cla1 {
  func baz() {}
  func bat(x: Int, y : Str1) -> Int { return 1 }
  var x = Str1()
  var y = Dictionary<Int,String>()
}

func main() {
  var s = Str1()
  var c = Cla1()
  print("break here")
}

class Toplevel {
	struct Nested {
		class Deeper {
			func foo() {}
		}
	}
}

class Generic<T> {
	func foo(x: T) {}
}

func foo(x: Int, y: Int) -> Int { return x + 1 }
func foo() -> Double { return 3.1415 }

main()
