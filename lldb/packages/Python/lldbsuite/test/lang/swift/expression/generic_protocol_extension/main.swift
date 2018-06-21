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

protocol P {
  var v : Int { get };
}

class C<T> : P {
  var v : Int { get { return 3 } };
} 

extension P {
  func f() -> Int {
    return v //% self.expect("p self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["a.C"])
  }
}

var c = C<Int>()
print(c.f())
