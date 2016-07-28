// modb.swift
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
import moda

extension S {
  private struct A {
    var v = 3
  }

  fileprivate func f() -> Int {
    let a = A()
    return a.v // break here
  }
}

public func fB(_ x: S) -> Int {
  return x.f()
}
