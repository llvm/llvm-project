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
  var path = IndexPath(indexes: [1,2,3,4,5])
  print("done!") //% self.expect("frame variable path", substrs=['5 indices'])
   //% self.expect("expression -d run -- path", substrs=['5 indices'])
}

main()
