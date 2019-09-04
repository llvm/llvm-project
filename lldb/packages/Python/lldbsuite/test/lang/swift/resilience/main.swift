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
import mod

func main() {
  initGlobal()
  let s = S()
  let a = fA(s)
  print(s.a) // break here
  print(a)
}

fileprivate class Message { fileprivate var s = "world" }
fileprivate struct NotBitwiseTakable {
  fileprivate weak var msg : Message?
}
private struct FixedContainer {
  var s = S()
}
private struct NestedFixedContainer {
  var s = FixedContainer()
}

private var g_main_msg = Message()
private var g_main_b = NotBitwiseTakable()
private var g_main_s = S()
private var g_main_t = (S(), S())
private var g_main_nested_t = ((1, S()), 2)
private var g_main_c = FixedContainer()
private var g_main_nested_c = NestedFixedContainer()
g_main_b.msg = g_main_msg
print(g_main_s.a + g_main_t.0.a + g_main_c.s.a)
main() // break here
