// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

import simd

func main() -> Int {
  let d4 = simd_double4(1.5, 2, 3, 4) //%self.expect('frame variable d4', substrs=['1.5', '2', '3', '4'])
  print(d4)
  return 0
}

main()
