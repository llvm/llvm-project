// tow.swift
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
private func == (lhs : Fooey, rhs : Fooey) -> Bool 
{ 
    Fooey.BumpCounter(2)
    return lhs.m_var != rhs.m_var // break here for two local operator
}

extension Fooey
{
    public class func CompareEm2(_ lhs : Fooey, _ rhs : Fooey) -> Bool 
    { 
        return lhs == rhs
    }
}

