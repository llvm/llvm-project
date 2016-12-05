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

func main()
{
	var s1 = "Hello world"
	var s2 = "ΞΕΛΛΘ"
	var s3 = s1 as NSString
	var s4 = s2 as NSString
	var s5 = "abc" as NSString
	var s6 = String(s5)
	print(s1) // Set breakpoint here
}

main()
