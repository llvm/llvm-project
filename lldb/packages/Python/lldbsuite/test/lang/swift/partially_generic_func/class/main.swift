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
class Generic<T> {
  let a = "Hello world"
  let b = 12
}

func foo<T0>(x: Generic<T0>) {
  print(x) //% self.expect('frame variable -d run -- x', substrs=['"Hello world"', '12'])
  //% self.expect('expression -d run -- x', substrs=['"Hello world"', '12'])
}

foo(Generic<Int>())
