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
class Inner {
  var data = "Hello World"
}

class Outer {
  var data = [Inner]()
  
  init() {
    var inner = Inner()
    data.append(inner)
    data.append(inner)
  }
}

func main() {
  var o = Outer()
  print(o) // break here
}

main()

