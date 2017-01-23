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
import Foundation

var my_global: Int = 0  // global variable

func main() {
  var thread1: pthread_t? = nil
  var thread2: pthread_t? = nil

  pthread_create(&thread1, nil, { _ in
    my_global = 42
    sleep(100)
    return nil
  }, nil)

  pthread_create(&thread2, nil, { _ in
    usleep(10000)
    my_global = 43
    exit(1)
  }, nil)
}

main()
