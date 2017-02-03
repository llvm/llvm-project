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
class PureSwift
{
    init (int_value: Int)
    {
        m_ivar = int_value
    }

    func stop_here ()
    {
        print ("Stop here in Pure Swift class")
    }

    var m_ivar : Int
    var m_computed_ivar : Int {
        get {
            return 5
        }
    }
}

var my_swift_object = PureSwift(int_value: 10)
my_swift_object.stop_here()
