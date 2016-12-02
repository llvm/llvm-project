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

class Base {
  func foo (_ a: Int!) {
    print(a)
  }
}

class Sub : Base {
  func foo (_ a: Int) {
    print(a+1)
  }
}

let b : Base = Sub()
b.foo(3) //% self.dbg.HandleCommand("thread step-in")
         //% self.expect("expr a", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["3"])
