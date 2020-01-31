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

private var a = 1

func doSomething(b: Int) {
	a += b //% self.expect("expr a", substrs=['Int', '= 1'])
}

func withLocalShadow() {
  let a = 23
  doSomething(b: a) //% self.expect("log enable lldb expr");self.expect("expr a", substrs=['Int', '= 23'])
}

withLocalShadow()
