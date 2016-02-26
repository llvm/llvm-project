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
import Foundation

class SwiftClass {
	var ns_a: NSString
	var sw_a: String
	var ns_d: NSData
	var sw_i: Int
	var ns_n: NSNumber
	var ns_u: NSURL
	
	init() {
		ns_a = "Hello Swift" as NSString
		sw_a = "Hello Swift"
		ns_d = NSData()
		sw_i = 30
		ns_n = 30 as NSNumber
		ns_u = NSURL(string: "http://www.apple.com")!
	}
}

func main() {
	var swcla = SwiftClass()
  var nsarr: NSArray = [2,3,"Hello",5,["One","Two","Three"]]
	print("hello world") // Set breakpoint here
}

main()

