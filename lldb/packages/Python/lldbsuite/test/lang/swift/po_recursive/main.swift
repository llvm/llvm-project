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
class A : CustomReflectable {
  var a: A?
  
  func customMirror() -> Mirror {
    return Mirror(self, children: ["a" : a], displayStyle: .Class)
  }
}

struct S {
  var a: A?
}

func main() {
  var s = S()
  s.a = A()
  s.a?.a = A()
  s.a?.a?.a = s.a
  print("a") //%self.expect('po s', substrs=['▿ a : Optional', 'Some', '0x', '{ ... }'])
}

main()
