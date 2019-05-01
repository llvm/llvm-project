// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

class ClosureMaker {
  var a : Int

  init (a : Int) {
    self.a = a
  }

  func getClosure() -> (() -> Int) {
    return { [weak self] () -> Int in
             if let _self = self {
               return _self.a //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker?)", "5"])
                              //% self.expect("expr self!", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker)", "5"])
             } else {
               return 0 //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["nil"])
            }
          }
  }

  func getGuardClosure() -> (() -> Int) {
    return { [weak self] () -> Int in
             guard let self = self else {
               return 0  //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["nil"])
             }
             return self.a //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker?)", "5"])
                          //% self.expect("expr self!", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker)", "5"])
          }
  }
}

func use<T>(_ t: T) {}

var livemaker : ClosureMaker? = ClosureMaker(a: 5)
let liveclosure = livemaker!.getClosure()
let liveguardclosure = livemaker!.getGuardClosure()
use((liveclosure(), liveguardclosure()))

var deadmaker : ClosureMaker? = ClosureMaker(a: 3)
let deadclosure = deadmaker!.getClosure()
let deadguardclosure = deadmaker!.getGuardClosure()
deadmaker = nil
use((deadclosure(), deadguardclosure()))
