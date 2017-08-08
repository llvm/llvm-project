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
import Foundation

class FooObjC : NSObject {
	var x = 12
	var y = "12"
}

class FooSwift {
	var x = 12
	var y = "12"
}

func main() {
	var foo_objc = FooObjC()
	var foo_swift = FooSwift()
	print("Set a breakpoint here") // Set a breakpoint here
}

main()
