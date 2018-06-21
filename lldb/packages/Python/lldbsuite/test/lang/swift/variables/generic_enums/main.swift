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
func foo <U> (_ myOptionalU : U?)
{
    var myU = myOptionalU! // Set first breakpoint here.
}

func log<T>(_ value : T) -> T {
    return value // Set second breakpoint here.
}

var i : Int? = 3

foo(i)

var missingString : String? = log("Now with Content")

print(missingString!)
