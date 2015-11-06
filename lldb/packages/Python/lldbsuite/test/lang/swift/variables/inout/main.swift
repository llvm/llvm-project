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
class Class {
    var ivar : Int
    init() 
    { 
        ivar = 1234 
    }
}

class Other : Class {
    var ovar : Int
    override init () 
    { 
        ovar = 112233
	super.init()
	ivar = 4321
    }
    init (in1 : Int, in2 : Int)
    {
        ovar = in2
        super.init()
        ivar = in1
    }
}

struct Struct {
	var ivar : Int
	init() { ivar = 4567 }
}

func foo (inout x : Class) {
	print(x.ivar)
	x.ivar++ // Set breakpoint here for Class access
}

func foo(inout x : Struct) {
	print(x.ivar)
	x.ivar++ // Set breakpoint here for Struct access
}

func main() {
	var x : Class = Other()
	var s = Struct()

	foo(&x)
	foo(&s)
	foo(&x)
        print ("Set breakpoint here after class access") 
	foo(&s)
}

main()

