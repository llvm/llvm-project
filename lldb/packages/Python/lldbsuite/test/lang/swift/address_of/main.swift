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
class Class {
    var c_x : UInt32 = 12345
    var c_y : UInt32 = 6789
}

struct Struct {
    var s_x : Int = 12345
    var s_y : Int = 6789
}

func takes_class(_ in_class: Class)
{
    print(in_class) // Break here in takes_class
}

func takes_struct(_ in_struct : Struct)
{
    print(in_struct) // Break here in takes_struct
}

func takes_inout(_ in_struct : inout Struct)
{
    print(in_struct) // Break here in takes_inout
}

func main() {
    var c = Class()
    var s = Struct()
    var int : Int = 100

    print("hello world") // Break here in main
    takes_class(c)
    takes_struct(s)
    takes_inout(&s)    
}

main()

