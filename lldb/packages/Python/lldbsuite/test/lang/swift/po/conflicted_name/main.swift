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
class Mirror : CustomDebugStringConvertible {
  func reflect<T>(x: T) -> T { return x }
  var debugDescription: String { return "Fun with mirrors" }
}

func main() {
  var m = Mirror()
  print("yay I am done!") //% self.expect("po m", substrs=['Fun with mirrors'])
}

main()
