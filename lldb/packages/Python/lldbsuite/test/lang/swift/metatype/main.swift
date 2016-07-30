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
class C {}
class D : C {}
protocol P {}

func main<T>(_ x: T) {
  var s = String.self
  var c = D.self
  var t = type(of: (1,2,"hello"))
  var p = P.self
  var f = T.self
  print("Set breakpoint here")
}

main() { (x:Int) -> Int in return x }
