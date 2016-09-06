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

func main() {
    var a: [String: Any] = [
        "Key1" : "Value1",
        "Key2" : [1234,5678],
        "Key3" : ["Object" as NSString, ["WAHHH" as NSString: 2467 as AnyObject] as NSDictionary] as NSArray
    ]
    
    print(a) //%self.expect('po a', substrs=['Key1','Value1','Key2','1234','5678','Object','WAHHH','2467'])
}

main()
