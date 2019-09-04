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
  // explicit type annotations here are by design - we want to make sure we bridge to types other than Int
  var nsarr: NSArray = NSArray(array: [Int(123456),Int32(234567),UInt16(45678),Double(1.25),Float(2.5)])
  var swarr = unsafeBitCast(unsafeBitCast(nsarr, to: Int.self), to: NSArray.self) as! [AnyObject]
  print("break here")
}

main()
