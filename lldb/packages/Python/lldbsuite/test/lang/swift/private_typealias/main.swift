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
public struct MyStruct {
    fileprivate typealias IntegerType = Int
    private var m_integer = IntegerType(345)
    fileprivate func Foo(_ i : IntegerType)
    {
        print("i = \(i)") // breakpoint 1
    }
    
    fileprivate func Bar()
    {
        let a : Dictionary<String, IntegerType> = [ "hello" : 234 ]
        print("a = \(a)") // breakpoint 2
    }
    
}

func main() {
    let s = MyStruct()
    s.Foo(123)
    s.Bar()
}

main()
