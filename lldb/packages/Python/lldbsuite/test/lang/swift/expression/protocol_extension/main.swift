// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
public protocol Foo
{
    func foo(x: Int) -> Int
}

struct FooishStruct : Foo
{
    func foo(x: Int) -> Int
    {
        return x + FooishStruct.cvar
    }
    
    let x = 10
    let y = "Hello world"
    let z = 3.14
    static let cvar = 333
}

class FooishClass : Foo
{
    func foo(x: Int) -> Int
    {
        return x + FooishStruct.cvar
    }
    
    let x = 10
    let y = "Hello world"
    let z = 3.14
    static let cvar = 333
}

extension Foo
{
    public static func bar()
    {
        let local_var = 222
        print("break here in static func \(local_var)")
    }
    
    public func baz()
    {
        let local_var = 111
        print("break here in method \(local_var)")
    }
}

func main()
{
    (FooishStruct()).baz()
    FooishStruct.bar()

    (FooishStruct()).baz()
    FooishStruct.bar()
}

main()
