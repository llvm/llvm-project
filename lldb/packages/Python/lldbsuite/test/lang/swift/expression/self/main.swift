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

struct StructTest {
  func foo() {
    print("Stop here in method \(m_var)") //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["StructTest"])
                                          //% self.expect("expr typealias foo = StructTest; self as foo", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["foo"]) 
  }
  let m_var = 234
}

class ClassTest {
  func foo () {
    print ("Stop here in method \(m_var)") //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClassTest"])
  }
  let m_var = 234
}

var st = StructTest()
st.foo()

var ct = ClassTest()
ct.foo()
