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


protocol AProtocol {
	associatedtype AType
	func makeOne() -> AType
}

class AClass : AProtocol {
	typealias AType = Int
	func makeOne() -> Int { return 1; }
	var ivar : Int
	init() { ivar = 0xDEADBEEF }
}

class AParentClass {
}

class ADerivedClass : AParentClass {
}

class AnotherDerivedClass : AParentClass {
}

func foo (_ x : AClass) {
	print("foo") // Set breakpoint here
}
func foo<T: AProtocol> (_ x : T) {
	print("foo") // Set breakpoint here
}
func bar <T,U> (_ x : T, _ y : U) {
	print("bar") // Set breakpoint here
}
func baz <T : AProtocol> (_ x : T) {
	print("baz") // Set breakpoint here
}
func bat (_ x : AParentClass) {
	print("bat") // Set breakpoint here
}

var aProtocol = AClass()
var aClass : AClass = AClass()
var aParentClass : AParentClass = ADerivedClass()
var anotherDerivedClass : AnotherDerivedClass = AnotherDerivedClass()

foo(aClass)
bar(1 as Int64,2.1 as Float32)
foo(aProtocol)
baz(aClass)
bar(aClass,aProtocol)
bat(aParentClass)
bat(anotherDerivedClass)
