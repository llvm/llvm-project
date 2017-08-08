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
}

var livemaker : ClosureMaker? = ClosureMaker(a: 5)
let liveclosure = livemaker!.getClosure()
print(liveclosure())

var deadmaker : ClosureMaker? = ClosureMaker(a: 3)
let deadclosure = deadmaker!.getClosure()
deadmaker = nil
print(deadclosure())
