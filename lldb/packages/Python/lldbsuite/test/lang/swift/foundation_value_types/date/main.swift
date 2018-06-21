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
  var date = Date(timeIntervalSince1970: 23*60*60)
  print("done!") //% self.expect("frame variable date", substrs=['1970-01-'])
   //% self.expect("expression -d run -- date", substrs=['1970-01-'])
}

main()
