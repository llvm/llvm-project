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
import Cocoa
import CoreData

func main() {
	var nss = "abc" as NSString
	var nsn = 3 as NSNumber
	var nsmo = CoreData.NSManagedObject()
	var nsmd = NSMutableDictionary()
	nsmd.setObject(nsn, forKey:nss)
	print("Hello world") // Set breakpoint here
}

main()
