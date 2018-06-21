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
struct Q<T> {
  let x: T
}
struct S<T> {
  init(x: [Q<T>]) {
    print(x) //% self.expect('frame variable -d run -- x', substrs=['1 value', '4276993775'])
  }
}

print(S(x: [Q(x: 0xFEEDBEEF)]))
