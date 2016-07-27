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
class MyString {
  private let content = "String"
}
  
class MyClass {
  fileprivate func processItem(_ item: MyString) -> () {
    debugPrint(item) //% self.expect("expr -d run -- self", substrs=['MyClass'])
  }
}

let i = MyString()
let o = MyClass()
o.processItem(i)
