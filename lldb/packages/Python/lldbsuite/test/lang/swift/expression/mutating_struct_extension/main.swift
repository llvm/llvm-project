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

extension Array
{
    mutating func doGenericStuff()
    {
        print("generic stuff") //% self.expect("expr 2+3", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["5"])
    }
}

var array = [Int]()
array.doGenericStuff()

