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
class Foo {
	var x: Int
  init(x: Int) { self.x = x }
}

struct LargeDude {
	var x : Int
	var y : Int
	var z : Int
	var q : Int
	
	init() { x = 0xFFFFFFFF; y = 0xDEADBEEF; z = 0xBEEFFEED; q=0x0FF00FF0; }
}

func main() {
	var arrint = Array<Int>()
	var arrfoo = Array<Foo>()
	var arrlar = Array<LargeDude>()
	
	arrint.append(1) // Set breakpoint here
	arrint.append(2)
	arrint.append(3)
	arrint.append(4)
	arrint.append(5)
    // arrint.reserve(10)
	
	arrfoo.append(Foo(x: 1)) // Set breakpoint here
	arrfoo.append(Foo(x: 2))
	arrfoo.append(Foo(x: 3))
	arrfoo.append(Foo(x: 4))
	arrfoo.append(Foo(x: 5))
	arrfoo.append(Foo(x: 6))
    // arrfoo.reserve(10)
	
	arrlar.append(LargeDude()) // Set breakpoint here
	arrlar.append(LargeDude())
	arrlar.append(LargeDude())
	arrlar.append(LargeDude())
	arrlar.append(LargeDude())
	arrlar.append(LargeDude())
	arrlar.append(LargeDude())
    // arrlar.reserve(12)

	var slice = arrint[1..<4]
	
	print("Hello world") // Set breakpoint here
	
}

main()
