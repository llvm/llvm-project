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
	let s1 = "Hello world"
	let s2 = "ΞΕΛΛΘ"
	let s3 = s1 as NSString
	let s4 = s2 as NSString
	let s5 = "abc" as NSString
	let s6 = String(s5)
	print(s1) // Set breakpoint here
}

main()
