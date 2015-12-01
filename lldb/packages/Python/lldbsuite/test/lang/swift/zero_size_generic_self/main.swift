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
struct GenericSelf<T> {
  init(x: T) {
    print(x) //%self.expect('frame variable -d run -- self', substrs=['GenericSelf<String>'])
  }
}

func main() {
  GenericSelf(x: "hello world")
}

main()
