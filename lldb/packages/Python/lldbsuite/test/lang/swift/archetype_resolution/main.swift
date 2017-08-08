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
func foo<T>(_ x: T) {
  print("break here")
}

struct S { 
  var x = 1
  var y = "hello"
  var z = 3.14
}

enum E {
  case A
  case B
  case C
}

class C {
  var x = 1
  var y = "hello"
  var z = 3.14
}

struct GS<T> {
  var x = 1
  var y = "hello"
  var z = 3.14
}

enum GE<T> {
  case A
  case B
  case C
}

struct GC<T> {
  var x = 1
  var y = "hello"
  var z = 3.14
}

func main() {
  foo("hello")
  foo(1)
  foo(S())
  foo(C())
  foo((1,2,3))
  foo(E.A)
  foo(GS<Int>())
  foo(GC<Int>())
  foo(GE<Int>.A)
}

main()
