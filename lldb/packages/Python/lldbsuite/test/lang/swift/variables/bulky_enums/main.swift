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
  var a = 1
  var b = 2
}

struct Q {
  var a = S()
  var b = S()
  
  init(a: Int, b: Int) {
    self.a = S(a: a, b: b)
    self.b = S(a: a, b: b)
  }
}

enum E {
  case A(Q,Q?)
  case B(Q,[Q]?)
  case C(Q,[Q]?)
  case D(Q,Q)
  case E(Q,Q)
  case F(Q,[Q]?)
  case G(Q,Q)
  case H(Q,Q)
  case I(Q,[Q]?)
  case J(Q,Q)
  case K(Q,Q)
  case L(Q,Q)
  case M(Q,Q)
  case N(Q,Q)
  case O(Q,Q)
  case P(Q,Q)
  case R(Q,Q)
  case T(Q,Q)
  case U(Q,Q)
  case V(Q,Q)
  case W(String,Q,Q?)
  case X(String,Q,Q)
  case Y(Q,Q?)
  case Z(Q,Q?)
}

func main() {
  var e: E? = E.X("hello world", Q(a: 100, b: 200), Q(a: 300, b: 400))
  print(e) // break here
}

main()
