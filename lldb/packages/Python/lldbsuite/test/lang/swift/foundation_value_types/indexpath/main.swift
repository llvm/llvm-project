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
  var short_path = IndexPath(indexes: [1,2])
  var very_short_path = IndexPath(indexes: [1])
  var empty_path = IndexPath()
  print("done!") //% self.expect("frame variable path", substrs=['5 indices'])
   //% self.expect("frame variable short_path", substrs=['2 indices'])
   //% self.expect("frame variable very_short_path", substrs=['1 index'])   
   //% self.expect("frame variable empty_path", substrs=['0 indices'])
   //% self.expect("expression -d run -- path", substrs=['5 indices'])
}

main()
