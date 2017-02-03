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
  var a: NSObject = 3 as NSNumber
  var b: AnyObject = 3 as NSNumber
  print("break here") //% self.expect('frame variable -d run -- a', substrs=['Int64(3)'])
   //% self.expect('frame variable -d run -- b', substrs=['Int64(3)'])
    //% self.expect('expr -d run -- a', substrs=['Int64(3)'])
     //% self.expect('expr -d run -- b', substrs=['Int64(3)'])
}

main()

