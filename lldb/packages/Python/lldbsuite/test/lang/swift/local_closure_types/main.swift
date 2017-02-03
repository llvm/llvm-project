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
let foo: () -> () = {
  struct S {
    let i: Int
  }
  let s = S(i: 777)
  print(s.i) //% self.expect("frame variable s", substrs=['i = 777'])
  //% self.expect("expr s", substrs=['i = 777'])
}

foo()
