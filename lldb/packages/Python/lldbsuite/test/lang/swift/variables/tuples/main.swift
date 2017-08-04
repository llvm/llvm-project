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
func getTuple() -> (length : Int, name : String)
{
    var t : (Int, String)
    t.0 = 123
    t.1 = "carp"
    return t
}

func getTuple2() -> (Int, String)
{
    var t : (Int, String)
    t.0 = 123
    t.1 = "carp"
    return t
}

func getTuple3() -> (Int, name : String)
{
    var t : (Int, String)
    t.0 = 123
    t.1 = "carp"
    return t
}

protocol Printable {
	func print()
}

extension String : Printable {
	func print() { Swift.print(self) }
}

func getGenericTuple <X : Printable,Y> (_ x : X, _ y : Y) -> (Y,X)
{
	var tuple = (y,x)
	tuple.1.print()   // Set breakpoint here
	return tuple
}

struct Point {
    var x: Float;
    var y: Float;
    init (_ _x: Float, _ _y: Float) {
        x = _x
        y = _y
    }
}

func getTuplePoints() -> (p1 : Point, p2 : Point)
{
    var p1 = Point(1.25, 2.125)
    var p2 = Point(4.50, 8.75 )
    var t = (p1, p2)
    return t
}

func main_function(_ t: (Int, Int, Int)) -> Int
{
	var tuple1 = getTuple() // Set breakpoint here
	var tuple2 = getTuple2()
	var tuple3 = getTuple3()
	var tuple4 = getTuplePoints()
	var tuple5 = getGenericTuple("hellow world",(1,2)) // Set breakpoint here
	return tuple1.0 + tuple2.0 + tuple3.0 + Int(tuple4.p2.y)
}

main_function((111, 222, 333))

