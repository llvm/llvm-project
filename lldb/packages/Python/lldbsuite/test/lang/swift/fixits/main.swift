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

class HasOptional
{
    var could_be : Int?
    init (input: Int)
    {
        could_be = input
    }

    init ()
    {
    }

    func returnIt () -> Int
    {
        if let value = could_be
        {
            return value
        }
        else
        {
            return 0
        }
    }
}

func main() -> Int 
{
    let does_have = HasOptional(input: 100)

    print ("\(does_have.returnIt()): break here to test fixits.")
    return 0
}

main()
