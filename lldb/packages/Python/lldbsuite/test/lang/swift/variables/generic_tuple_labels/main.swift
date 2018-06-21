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
func foo<T>(_ x: T) -> Int {
  print(x)
  return 42 // break here
}

func main() {
  let tuple = (x: 12, 24, z: 36, 48, q: 60, w: 72)
  foo(tuple)
  print(tuple)
}

main()
