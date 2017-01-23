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

func main() {
  var uuid = UUID(uuidString: "AE5DE240-397B-4D09-9B99-D38E4CBC9952")!
  print("done!") //% self.expect("frame variable uuid", substrs=['AE5DE240-397B-4D09-9B99-D38E4CBC9952'])
   //% self.expect("expression -d run -- uuid", substrs=['AE5DE240-397B-4D09-9B99-D38E4CBC9952'])
}

main()
