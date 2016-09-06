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
import Foundation

class Test: NSArray { 
    override var count: Int { return 1 } 
    override func object(at index: Int) -> Any { return "abc" } 
    override func copy(with: NSZone?) -> Any { return self } 
}

func main() {
  var t = Test()
  var ta = Test() as Array
  var tb = Test() as Array + []

  print("second stop") //% self.expect('frame variable -d run -- t', substrs=['t = 0x', 'NSArray = {', 'NSObject = {'])
                       //% self.expect('frame variable -d run -- ta', substrs=['ta = {', '_buffer = {', '_storage =', 'rawValue = 0x'])
                       //% self.expect('frame variable -d run -- tb', substrs=['tb = 1 value {', '[0] = "abc"'])
                       //% self.expect('po t', substrs=['0 : abc'])
                       //% self.expect('po ta', substrs=['0 : abc'])
                       //% self.expect('po tb', substrs=['0 : abc'])
}

main()
