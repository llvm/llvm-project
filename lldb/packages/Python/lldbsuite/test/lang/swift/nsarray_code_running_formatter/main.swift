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
import Cocoa

class Person: NSObject {
    var array = [AnyObject]()
}

func main() -> Int {
  let person = Person()
  let array = person.mutableArrayValueForKey("array")

  array.addObject(3)
  array.addObject(4)
  array.addObject(5)
  
  return 6 //% self.expect("expr -d run -- array", substrs=['Int64(3)','Int64(4)','Int64(5)'])
           //% self.expect("frame variable -d run -- array", substrs=['Int64(3)','Int64(4)','Int64(5)'])
           //% self.process().Kill()
}

main()
