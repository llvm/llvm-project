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

// Class, function

class B
{
  class func b (i : Int)
  {
    var x = i // breakpoint 1
  }
}

// Generic class, function

class C<T>
{
  class func c (i : T)
  {
    var x = i // breakpoint 2
  }
}

// Class, generic function

class E
{
  class func e<U> (i : U)
  {
    var x = i // breakpoint 3
  }
}

// Generic class, generic function

class F<T>
{
  class func f<U> (i : T, _ j : U)
  {
    var x = i
    var y = j // breakpoint 4
  }
}

struct G
{
  static func g(i : Int)
  {
    var x = i // breakpoint 5    
  }
}

struct H<T>
{
  static func h(i : T)
  {
    var x = i // breakpoint 6
  } 
}

struct K<T>
{
  static func k<U> (i: T, _ j : U)
  {
    var x = i
    var y = j  // breakpoint 7 
  }
}


// Invocations

B.b(1)

C<Int>.c(2)

E.e (3)

F<Int>.f (4, 10.1)

G.g(5)

H<Int>.h(6)

K<Int>.k(7,8.0)


