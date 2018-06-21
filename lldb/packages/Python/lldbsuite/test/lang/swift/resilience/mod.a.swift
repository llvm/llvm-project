// mod.a.swift
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
  public var a = 1
  public var s1 = "i"
  public var s2 = "am"
  public var s3 = "large"
  public var s4 = "!"

  public init() {}
}

public func fA(_ x: S) -> Int {
  return x.a
}
