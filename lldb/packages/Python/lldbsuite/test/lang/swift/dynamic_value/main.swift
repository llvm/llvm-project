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


class SomeClass {
	var x : Int
	
	init() {
		x = 33
	}
}

class SomeOtherClass : SomeClass {
	var y : Int
	
	override init () {
		y = 0xFF
	    super.init()
		x = 66
	}
}

class YetAnotherClass : SomeOtherClass {
	var z : Int
	
	override init() {
		z = 0xBEEF
	    super.init()
		x = 99
		y = 0xDEAD
	}
}

class AWrapperClass {
	var aWrappedObject : SomeClass
	
	init() {
		aWrappedObject = YetAnotherClass()
	}
}

class Base <A> {
	var v : Int
	
	init(_ x : A) {
	    v = 0x1ACABA1A 
	}
}

class Derived <A> : Base<A> {
	var q : Int
	override init (_ x : A) {
        q = 0xDEADBEEF
        super.init(x)  
	}
}

func app() -> () {
	var aWrapper = AWrapperClass()
	var anItem : SomeClass = aWrapper.aWrappedObject
	var aBase : Base<Int> = Derived(3)
	print("a fooBar is a cute object, too bad there is no such thing as a fooBar in this app") // Set a breakpoint here
	print("this line is pretty much useless, just used to step and see if we survive a step") // Set a breakpoint here
}

app()
