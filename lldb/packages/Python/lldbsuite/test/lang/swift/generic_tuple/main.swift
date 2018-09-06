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
func use<T>(_ t : T) {}

func single<T>(_ t : T) {
  let x = t
  use(x) //% self.expect('expr -d run-target -- t', substrs=['hello'])
         //% self.expect('expr -d run-target -- x', substrs=['hello'])
         //% self.expect('fr var -d run-target t', substrs=['String', 'hello'])
         //% self.expect('fr var -d run-target x', substrs=['hello'])
}

func string_tuple<T, U>(_ t : (T, U)) {
  let (_, y) = t
  use(y) //% self.expect('expr -d run-target -- t', substrs=['hello', 'hello'])
         //% self.expect('expr -d run-target -- y', substrs=['hello'])
         //% self.expect('fr var -d run-target t',
         //%             substrs=['(String, String)', 'hello', 'hello'])
         //% self.expect('fr var -d run-target y', substrs=['hello'])
}

func int_tuple<T, U>(_ t : (T, U)) {
  let (_, y) = t
  use(y) //% self.expect('expr -d run-target -- t',
         //%             substrs=['(Int32, Int64)', '111', '222'])
         //% self.expect('expr -d run-target -- y', substrs=['222'])
         //% self.expect('fr var -d run-target t',
         //%             substrs=['(Int32, Int64)', '111', '222'])
         //% self.expect('fr var -d run-target y', substrs=['222'])
}

let s = "hello"
single(s)
string_tuple((s, s))
int_tuple((Int32(111), Int64(222)))
