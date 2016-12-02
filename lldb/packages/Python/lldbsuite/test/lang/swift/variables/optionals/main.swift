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
struct S {
  var a : Int = 12
  var b : String = "Hello world"
}

func getSome<T> (_ x : T) -> T?
{
  return x
}

func getUncheckedSome<T>(_ x : T) -> T!
{
  return x
}

func main() {
  var optS_Some = getSome(S())
  var uoptS_Some = getUncheckedSome(S())

  var optString_Some = getSome("hello")
  var uoptString_Some = getUncheckedSome("hello")

  var optS_None : S? = nil
  var uoptS_None : S! = nil

  var optString_None : String? = nil
  var uoptString_None : String! = nil

  print("//Set breakpoint here") // Set breakpoint here
}

main()
