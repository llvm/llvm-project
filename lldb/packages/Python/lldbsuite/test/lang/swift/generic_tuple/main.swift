// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
func single<T>(_ t : T) -> T {
  let x = t
  return x //% self.expect('expr   t', substrs=['hello'])
           //% self.expect('fr var t', substrs=['hello'])
           //% self.expect('expr   x', substrs=['hello'])
           //% self.expect('fr var x', substrs=['hello'])
}

func tuple<T, U>(_ t : (T, U)) -> U {
  let (_, y) = t
  return y //% self.expect('expr   t', substrs=['hello', 'hello'])
           //% self.expect('fr var t', substrs=['hello', 'hello'])
           //% self.expect('expr   y', substrs=['hello'])
           //% self.expect('fr var y', substrs=['hello'])
}

let s = "hello"
print(single(s))
print(tuple((s, s)))
