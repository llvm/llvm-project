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
  var data = Data(bytes: [1,2,3,4,5])
  print("break here") //% self.expect('frame variable data', substrs=['5 bytes'])
   //% self.expect('expression data', substrs=['5 bytes'])
  data.append(data)
  print("break here") //% self.expect('frame variable data', substrs=['10 bytes'])
  //% self.expect('expression data', substrs=['10 bytes'])
  //% self.expect('expression data.subdata(in: Range(uncheckedBounds: (lower: data.startIndex, upper: data.index(after: data.startIndex))))', substrs=['1 byte'])
}

main()
