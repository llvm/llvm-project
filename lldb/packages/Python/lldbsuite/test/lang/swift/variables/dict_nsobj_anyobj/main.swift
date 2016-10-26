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

func main() {
  var d1: Dictionary<Int,Int> = [1:1,2:2,3:3,4:4]
  var d2: Dictionary<NSObject,AnyObject> = [1 as NSNumber : 1 as NSNumber,
                                            2 as NSNumber : 2 as NSNumber,
                                            3 as NSNumber : 3 as NSNumber,
                                            4 as NSNumber : 4 as NSNumber]
  var d3: AnyObject = Dictionary<String, Int>(dictionaryLiteral: ("hello", 123)) as NSDictionary
  print("break here")
}

main()
