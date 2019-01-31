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
extension Array {
    func test() {
        print("PATPAT") //% self.expect("p self", substrs=['[0] = 1', '[1] = 2', '[2] = 3'])
        return //% self.expect("frame variable -d run -- self", substrs=['[0] = 1', '[1] = 2', '[2] = 3'])
    }
}
var a = [1,2,3]

a.test()
