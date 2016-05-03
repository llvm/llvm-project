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
import Foundation

var my_global: Int = 0

func main() {
  let q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0)
  dispatch_async(q) {
    my_global = 42
    sleep(100)
  }
  dispatch_async(q) {
    usleep(10000)
    my_global = 43
    exit(1)
  }
  sleep(100)
}

main()
