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
#if os(iOS)
    import UIKit
#elseif os(OSX)
    import AppKit
#endif    

func main() {
  var num = 22
  //% self.expect("po num", substrs=["\\n", "\""], matching=False) # Make sure po doesn't escape non-printables.
  var str = "Hello world" //% self.expect("po num", substrs = ['22'])
  var arr = [1,2,3,4] 
  var nsarr = NSMutableArray(array: arr) //% self.expect("po str", substrs = ['Hello world'])
#if os(iOS)
  var clr = UIColor.red //% self.expect("po arr", substrs = ['1','2','3','4'])
#elseif os(OSX)
  var clr = NSColor.red //% self.expect("po arr", substrs = ['1','2','3','4'])
#endif
  //% self.expect("po nsarr", substrs = ['1','2','3','4'])
  var nsobject = NSObject() //% self.expect("po clr", substrs = ['1 0 0 1']) # may change depending on OS/platform
  var any: Any = 1234 //% self.expect("po nsobject", substrs = ['<NSObject: 0x']) # may change depending on OS/platform
  //% self.expect("script lldb.frame.FindVariable('nsobject').GetObjectDescription()", substrs = ['<NSObject: 0x']) # may change depending on OS/platform
  var anyobject: AnyObject = 1234 as NSNumber //% self.expect("po any", substrs = ['1234'])
  var notification = Notification(name: Notification.Name(rawValue: "JustANotification"), object: nil)
  print("yay I am done!") //% self.expect("po notification", substrs=['JustANotification'])
   //% self.expect("po notification", matching=False, substrs=['super'])
}

main()
