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
public protocol Foo
{
    func foo(_ x: Int) -> Int
}

struct FooishStruct : Foo
{
    func foo(_ x: Int) -> Int
    {
        return x + FooishStruct.cvar
    }
    
    let x = 10
    let y = "Hello world"

    static let cvar = 333
}

class FooishClass : Foo
{
    func foo(_ x: Int) -> Int
    {
        return x + FooishStruct.cvar
    }
    
    let x = 10
    let y = "Hello world"

    static let cvar = 333
}

enum FooishEnum : Int, Foo
{
    case One = 1
    case Two = 2

    var x : Int {return 10}
    var y : String { return "Hello world"}

    func foo(_ x: Int) -> Int
    {
        return x + FooishEnum.cvar
    }
    
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

    FooishEnum.One.baz()
    FooishEnum.bar()
}

main()
