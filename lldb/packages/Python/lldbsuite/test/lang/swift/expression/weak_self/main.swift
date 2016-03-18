//===-- main.swift --------------------------------------------*- Swift -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

class ClosureMaker {
    var a : Int

    init (a : Int) {
        self.a = a
    }

    func getClosure() -> (() -> Int) {
        return { [weak self] () -> Int in
            if let _self = self { //% self.expect("expr self", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["ClosureMaker"])
                return _self.a
            } else {
                return 0
            }
        }
    }
}

var maker : ClosureMaker? = ClosureMaker(a: 5)

let closure = maker!.getClosure()

print(closure())
