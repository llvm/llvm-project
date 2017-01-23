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

struct S {
  var a: Int8 = 1
  var b: Int8 = 1
  var c: Int8 = 1
  var d: Int8 = 1
}

func main() {
  var s: S? = S()
  print(s) // break here
}

main()
