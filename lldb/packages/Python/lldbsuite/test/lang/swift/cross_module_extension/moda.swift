// moda.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
public struct S {
  var a = 1
  private var b = 2
  
  private struct A {
    var v = 1
  }
  
  fileprivate func f() -> Int {
    let a = A()
    return a.v // break here
  }
  
  public init() {}
}

public func fA(_ x: S) -> Int {
  return x.f()
}
