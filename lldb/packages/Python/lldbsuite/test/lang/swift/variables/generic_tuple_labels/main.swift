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
func foo<T>(x: T) -> Int {
  print(x)
  return 42 // break here
}

func main() {
  let tuple = (x: 12, 24, z: 36, 48, q: 60, w: 72)
  foo(tuple)
  print(tuple)
}

main()
