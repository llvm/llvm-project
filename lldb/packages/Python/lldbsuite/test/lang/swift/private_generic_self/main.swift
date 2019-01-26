// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
protocol MyKey {}
protocol MyProtocol {
  associatedtype Key : MyKey
}
struct MyStruct<S : MyKey> : MyProtocol {
  typealias Key = S
}
extension MyProtocol {
  func decode() {
        return //% lldbutil.check_variable(self, self.frame().FindVariable("self"),use_dynamic=True,use_synthetic=True,typename='a.MyStruct<a.Outer.CodingKeys>')
    }
}

struct Decoder {
  func container<T : MyKey>(keyedBy type : T.Type) -> MyStruct<T> {
    return MyStruct<T>()
  }
}

struct Outer {
  private enum CodingKeys: String, MyKey {
    case features
  }
  init(from decoder: Decoder) throws {
    let container = decoder.container(keyedBy: CodingKeys.self)
    container.decode()
  }
}

let _ = try Outer(from: Decoder())
