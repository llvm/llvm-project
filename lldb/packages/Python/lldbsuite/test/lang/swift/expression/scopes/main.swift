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
class A
{
    var in_class_a : Int
    var also_in_a : Int
    var backing_int : Int = 10
    let m_char = "a"

    var computed_int : Int
    {
        get 
        {
            print ("In get.")
            return backing_int
        }
        set (new_value)
        {
            print ("In set.")
            backing_int += new_value + 1
        }
    }

    class func static_method (_ input : Int)
    {
        print("Got input: \(input) and ten: \(self.return_ten())") // In class function
    }

    class func return_ten () -> Int
    {
        return 10
    }

    subscript (input : Int) -> Int
    {
        get
        {
            return input * 5  // In int subscript getter
        }

        set (new_value)
        {
            print ("I think not.") // In int subscript setter
        }
    }

    subscript (input : String) -> Int
    {
        get
        {
            return input.count  // In string subscript getter
        }

        set (new_value)
        {
            print ("I like my value better.")  // In string subscript setter
        }
    }

    init (_ input : Int)
    {
        in_class_a = input
        also_in_a = input + 1
        print ("In init.")
    }

    init (_ input: String)
    {
        in_class_a = input.count
        also_in_a = in_class_a + 1
        print ("In string init.")
    }

    deinit
    {
        print ("In deinit.")
    }

    func shadow_a (_ input : Int) -> Void
    {
        var in_class_a = 10.0
        print("Got input: \(input) shadower: \(in_class_a) shadowed: \(self.in_class_a) and also_in_a: \(also_in_a) and through self: \(self.also_in_a)") // Shadowed in A
    }

    func DefinesClosure (_ a_string : String) -> () -> String
    {
        return { [unowned self] in
                  var tmp_string = a_string 
                  tmp_string += self.m_char // Break here in closure
                  return tmp_string
               }
    }
}

struct B
{
    var a : String = "foo"
    var b : Int = 3
    mutating func method() -> Void {
        print(a)
        b = 5
        print(b) // Break here in struct
    }
}

func main () -> Void
{
    var my_a = A(20)
    my_a.shadow_a (30)
    A.static_method(10)
    my_a.computed_int = 30
    var foo = my_a.computed_int
    my_a = A("A thirty character long string")

    var sub_int_int : Int = my_a[30]
    my_a[30] = 10

    var sub_str_int : Int = my_a["Some string"]
    my_a["Some string"] = 10 
    var some_string = my_a.DefinesClosure("abcde")()

    var my_b = B()

    my_b.method()
}

main()
