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
public final class PerformanceMetric {
   public let label: String = "label"
}

public struct Continuation<A> {
   private let magicToken = "Hello World"
   fileprivate let f: (() -> A)?
   fileprivate let failable: (() -> Continuation<A>?)?
   fileprivate var perfMetric: PerformanceMetric?

   public func run() -> A? {
       if let ff = f {
           return ff()
       } else if let ff = failable {
           return ff()?.run()
       } else {
           return nil
       }
   }
}

public typealias ContinuationU = Continuation<()>

public func sequence_<A>(_ xs: [Continuation<A>]) -> ContinuationU {
   return ContinuationU(f: nil, failable: {
       for x in xs {
           if x.run() != nil { //% self.expect('frame variable -d run -- x', substrs=['magicToken = "Hello World"', 'f = nil', 'failable = nil', 'perfMetric = nil'])
                               //% self.expect('expression -d run -- x', substrs=['magicToken = "Hello World"', 'f = nil', 'failable = nil', 'perfMetric = nil'])
           } else {
               return nil
           }
       }
       return nil
   }, perfMetric: nil)
}

var cont = Continuation<Void>(f: nil, failable: nil, perfMetric: nil)

print(cont)

sequence_([cont]).run()
