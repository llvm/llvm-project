// main.swift
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
struct HasInt {
  var value: Int
  
  init(_ x: Int) {
    value = x
  }
}

typealias Foo = HasInt
typealias Bar = Foo

func main() {
  var f: Foo = HasInt(12)
  var b: Bar = HasInt(24)
  print("break here")
}

main()
