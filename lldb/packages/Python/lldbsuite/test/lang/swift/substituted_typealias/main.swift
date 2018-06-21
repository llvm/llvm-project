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
typealias X = Int

func main() {
  let d = unsafeBitCast(5, to: Int.self)
  let x = unsafeBitCast(5, to: X.self)
  print("break here and do test") //%self.expect('frame variable d', substrs=['5'])
  //%self.expect('frame variable x', substrs=['(X)', '5'])
}

main()
