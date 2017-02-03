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


func foo(_ x: Int, _ y: Int) -> Int {
  return x - y + 1 // Set breakpoint here
}

foo(1,4)
foo(5,1)
foo(5,5)
foo(3,-1)
foo(6,6)
foo(7,7)
foo(1,3)
foo(3,1)

