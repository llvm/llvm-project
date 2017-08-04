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
import Foundation

func main() {
  var measurement = Measurement(value: 1.25, unit: Unit(symbol: "m"))
  print("done!") //% self.expect("frame variable measurement", substrs=['1.25 m'])
   //% self.expect("expression -d run -- measurement", substrs=['1.25 m'])
}

main()
