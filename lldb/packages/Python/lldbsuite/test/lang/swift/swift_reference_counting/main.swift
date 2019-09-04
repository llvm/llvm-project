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
class Patatino {
  var meh : Int

  init(_ meh : Int) {
    self.meh = meh
  }
}

struct Tinky {}

func lambda(_ Arg : Patatino) -> Int {
  return Arg.meh
}

func main() -> Int {
  var LiveObj = Patatino(37) //%self.expect('language swift refcount Blah', substrs=['unresolved identifier \'Blah\''], error=True)
  var Ret : Int = lambda(LiveObj) //%self.expect('language swift refcount LiveObj', substrs=['(strong = 3, unowned = 1, weak = 1)'])
  var MyStruct = Tinky() //%self.expect('language swift refcount MyStruct', substrs=['refcount only available for class types'], error=True)
  return Ret
}

main()
