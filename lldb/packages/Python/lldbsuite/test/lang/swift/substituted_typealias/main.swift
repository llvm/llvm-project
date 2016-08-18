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
typealias X = Double

func main() {
  let d = unsafeBitCast(5, to: Double.self)
  let x = unsafeBitCast(5, to: X.self)
  print("break here and do test") //%self.expect('frame variable d', substrs=['2.'])
  //%self.expect('frame variable x', substrs=['(X)', '2.'])
  //%self.expect('frame variable d', substrs=['_value'], matching=False)
  //%self.expect('frame variable x', substrs=['_value'], matching=False)
}

main()
