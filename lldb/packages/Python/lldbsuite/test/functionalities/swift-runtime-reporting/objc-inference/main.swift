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

class MyClass : NSObject {
  func memberfunc() { }  // method line
  func memberfunc2() { }  // method2 line
}

let mc = MyClass()
mc.perform(Selector(String("memberfunc")))
mc.perform(Selector(String("memberfunc2")))
