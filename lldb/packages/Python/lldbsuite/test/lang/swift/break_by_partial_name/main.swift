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


class Accumulator {
	var _total : Int
	init () {
		_total = 0
	}
	
	func incr (_ x : Int) -> (Int, Int) {
		var ret = (_total,0)
		_total += x
		ret.1 = _total
		return ret
	}
	
	func decr (_ x : Int) -> (Int, Int) {
		return incr(-x)
	}
	
	func getTotal() -> Int {
		return incr(0).0
	}
}

var x = Accumulator()
x.incr(5)
x.decr(2)
print(x.getTotal())
