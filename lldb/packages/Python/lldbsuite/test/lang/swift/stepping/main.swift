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
class ClassA
{
    var x : Int
    var y : Float

    init (_ input : Int)                 // Here is the A constructor.
    {
        x = input                     // A init: x assignment.
        y = Float(input) + 0.5        // A init: y assignment.
    }                                 // A init: end of function.
    init (_ input : Float)
    {
        x = Int(input)
        y = input
    }

    func do_something (_ input: Int) -> Int
    {
        if (input > 0)   // A.do_something - step in should stop here
        {
            return x * input
        }
        else
        { 
            return -x * input
        }
    }
}

class ClassB : ClassA
{
    override init (_ input : Int) // At the first line of B constructor.
    {
        super.init (input)  // In the B constructor about to call super.
    }

    override func do_something (_ input: Int) -> Int
    {
        var decider : Bool
        decider = input % 2 == 0
        if decider
        {
            return super.do_something(input)  // B.do_something: Step into super from here
        }
        else
        {
            return 0
        }
    }
}

func call_overridden (_ class_a_object : ClassA, _ int_arg : Int) -> Int  // call_overridden func def
{
    return class_a_object.do_something(int_arg)
}

enum SomeValues : Equatable 
{
    case Five 
    case Six 
    case Eleven
    case AnyValue
}

func == (lhs: SomeValues, rhs: SomeValues) -> Bool
{
    switch (lhs, rhs)
    {
        case (.Five, .Five),
             (.Six, .Six),
             (.Eleven, .Eleven):
            return true
        case (_, _):
            return false
    }
}

extension Int
{
    func toSomeValues() -> SomeValues
    {
        switch self
        {
            case 5:
                return .Five
            case 6:
                return .Six
            case 11:
                return .Eleven
            case _:
                return .AnyValue
        }
    }
}

extension SomeValues 
{
    func toInt() -> Int
    {
        switch self
        {
        case .Five:
            return 5
        case .Six:
            return 6
        case .Eleven:
            return 11
        case .AnyValue:
            return 46
        }
    }
}

protocol P 
{ 
    func protocol_func(_ arg : Int) -> Int
}

class ConformsDirectly : P 
{
    var m_value : Int

    init()
    {
        m_value = 64
    }

    init (_ value : Int)
    {
        m_value = value
    }

    func protocol_func(_ actual_arg : Int) -> Int  // We stopped at the protocol_func declaration instead.
    { 
        print("protocol_func(Int) from A: \(actual_arg).") // This is where we will stop in the protocol dispatch
        return m_value + actual_arg
    }

}

class ConformsIndirectly : ConformsDirectly 
{
    override init ()
    {
        super.init(32)
    }
}

func main () -> Void
{
    var some_values : SomeValues = 5.toSomeValues() // Stop here first in main
    var other_values : SomeValues = .Eleven

    if some_values == other_values // Stop here to get into equality
    {
        print ("I should not get here.")
    }
    else
    {
        print ("I should get here.") // Step over the if should get here
    } // Step over the print should get here.

    var b_object = ClassB(20)  // Stop here to step into B constructor.
    var do_something_result = call_overridden (b_object, 30) // Stop here to step into call_overridden.

    var point = (1, -1)  // At point initializer.
    func return_same (_ input : Int) -> Int
    {
        return input; // return_same gets called in both where statements
    }

    switch point  // At the beginning of the switch. 
    {
        case (0, 0):
            print("(0, 0) is at the origin")
        case (_, 0):
            print("(\(point.0), 0) is on the x-axis")
        case (0, _):
            print("(0, \(point.1)) is on the y-axis")
        case (let x, let y) where  
                return_same(x) == return_same(y): // First case with a where statement.
            print("(\(x), \(y)) is on the line x == y")
        case (let x, let y) where // Sometimes the line table steps to here after the body of the case. 
                return_same(x) == -return_same(y): // Second case with a where statement.
            print("(\(x), \(y)) is on the line x == -y") // print in second case with where statement.
        case (let x, let y):
            print("Position is: (\(x), \(y))")
    }  // This is the end of the switch statement

    var direct : P = ConformsDirectly() // Make a version of P that conforms directly
    direct.protocol_func(10)
    
    var indirect : P = ConformsIndirectly() // Make a version of P that conforms through a subclass
    indirect.protocol_func(20)

    var cd_maker = 
    { 
        (arg : Int) -> ConformsDirectly in  // Step into cd_maker stops at closure decl instead.
            return ConformsDirectly(arg) // Step into should stop here in closure.
    }

    func doSomethingWithFunction<Result : P> (_ f: (_ arg : Int)->Result, _ other_value : Int) -> Result // Stopped in doSomethingWithFunctionResult decl. 
    {
        print("Calling doSomethingWithFunction with value \(other_value)")
        let result = f(other_value)
        result.protocol_func(other_value)
        print ("Done calling doSomethingWithFunction.")
        return result
    }

    doSomethingWithFunction(cd_maker, 10)

}

main()
