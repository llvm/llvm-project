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
class BlubbyUbby<T>
{
    var my_int : Int
    var my_string : String
    var my_t : T
    
    init(_ in_int: Int, _ in_string : String, _ in_t : T)
    {
        my_int = in_int
        my_string = in_string
        my_t = in_t
    } //% self.expect('expr -d run -f hex -- my_t', substrs=['deadbeef'])
    //% self.expect('fr var -d run -f hex -- self.my_t', substrs=['deadbeef'])
    //% self.expect('expr -d run -- self', substrs=['3735928559'])
    //% self.expect('fr var -d run -- self', substrs=['3735928559'])

    func change_string (_ in_string : String)
    {
        my_string = in_string
    }

    func change_t (_ in_t : T)
    {
        my_t = in_t
    }
}

func main()
{
    var my_blubby = BlubbyUbby<Int>(1, "some string", 0xDeadBeef)
    my_blubby.change_t(3)
    print (my_blubby)
}

main()
