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
  // Data's .empty case
  var data = Data()
  print("break here") //% self.expect('frame variable data', substrs=['0 bytes'])
   //% self.expect('expression data', substrs=['0 bytes'])

  // Data's .inline case
  data.append(Data([1,2,3]))
  print("break here") //% self.expect('frame variable data', substrs=['3 bytes'])
   //% self.expect('expression data', substrs=['3 bytes'])
   //% self.expect('expression data.subdata(in: data.startIndex ..< data.index(after: data.startIndex))', substrs=['1 byte'])

  // Data's .slice case
  data.append(Data(repeating: 0xFF, count: 256))
  print("break here") //% self.expect('frame variable data', substrs=['259 bytes'])
   //% self.expect('expression data', substrs=['259 bytes'])
   //% self.expect('expression data.subdata(in: data.startIndex ..< data.index(after: data.startIndex))', substrs=['1 byte'])

  // NOTE: Data's .large case requires a UInt32.max-sized allocation on 64-bit,
  //       and a UInt16.max-sized allocation on 32-bit. Such a large allocation
  //       is likely not worth testing and risking destabilizing the test in
  //       memory-constrained scenarios.
}

main()
