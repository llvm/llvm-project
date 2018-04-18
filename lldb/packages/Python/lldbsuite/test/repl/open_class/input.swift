// input.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

// FIXME: There's no reason this can't be a lit-based REPL test, instead of a
// Python test built on 'expr', but at the time the test was written the REPL
// lit tests were not working on the contributor's machine.

class Foo {
  // Don't make any of these 'open'.
  typealias X = Int
  init() {}
  let value = 1
  final func foo() -> Int { return 2 }
}
///
Foo.X()
///= 0
Foo().value
///= 1
Foo().foo()
///= 2
class Bar: Foo {}
///
Bar().foo()
///= 2
class Baz: Foo {
  // Make sure 'foo' is still 'final'
  override func foo() -> Int { return 4 }
}
/// <could not resolve type>
