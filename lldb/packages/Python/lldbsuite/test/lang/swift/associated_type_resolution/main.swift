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
protocol Foo {
  associatedtype T
}

extension Foo {
  func foo(_ things: [T]) {
    // Set a breakpoint here
    for thing in things { print(thing) }
  }
}

struct Test: Foo {
  typealias T = Int
}

func main() {
  Test().foo([0,1,2,3])
}

main()
