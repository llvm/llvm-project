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
func getDict() -> Dictionary<Int, String>
{
    var d = Dictionary<Int, String>()
    d[12] = "hello"
    d[42] = "the number"
    return d
}

// Test bound generic structs
func getOptionalString() -> Optional<String>
{
    var opt_str = Optional<String>.some("Hello")
    return opt_str
}


// This will test our ability to create generics inside of classes
protocol Creatable {
	static func create() -> Self
	func print()
}

extension Int : Creatable {
	static func create() -> Int {
		return 255
	}
	func print() {
		Swift.print(self)
	}
}

func printLocal <JustSomeType : Creatable> () -> JustSomeType {
	var object = JustSomeType.create()
	object.print()  // Set breakpoint here, verify dynamic value of object
	return object
}


func main_function() {
    var d = getDict();
    var o_some = getOptionalString()
    var o_none = Optional<String>.none
    var c : Int = printLocal()
    print("stop here \(c)") // Set breakpoint here, verify values of "c", "d", and "o"
}

main_function()

