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
public struct Q<T> {
  let x: T
}
public func foo<T>(arg: [Q<T>]) {
  print(arg) //% self.expect('po arg', substrs=['x : 3735928559'])
  //% self.expect('expr -d run -- arg', substrs=['x = 3735928559'])
  //% self.expect('frame var -d run -- arg', substrs=['x = 3735928559'])
}

foo([Q(x: 0xdeadbeef)])
