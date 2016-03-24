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

class MyClass : NSObject {
  var text: String
  
  init(_ text: String) {
    self.text = text
  }
}

func main() {
  var cls: MyClass = MyClass("Instance of MyClass")
  var any: AnyObject = cls
  var opt: AnyObject? = cls
  var dict: [String: AnyObject] = ["One" : MyClass("Instance One"), "Two" : MyClass("Instance Two"), "Three" : cls]
  print(cls) // break here
}

main()
