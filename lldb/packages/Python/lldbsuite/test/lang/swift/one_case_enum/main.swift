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
protocol Event {
    
}

enum SillyEvent : Event {
    case Goofus
}

func doStuff<T>(event: T) -> T {
    return event // Set breakpoint here
}

func main() {
  var event: Event = SillyEvent.Goofus
  doStuff(event) // Set breakpoint here
}

main()
