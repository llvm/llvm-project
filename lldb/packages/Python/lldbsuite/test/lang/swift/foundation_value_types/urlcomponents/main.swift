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

func main() {
  var urlc = URLComponents(string: "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake")!
  print(urlc.scheme) //% self.expect('frame variable -d run -- urlc', substrs=['urlString = "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake"'])
  print(urlc.host) //% self.expect('frame variable -d run --  urlc', substrs=['schemeComponent = "https"'])
  print(urlc.port) //% self.expect('frame variable -d run --  urlc', substrs=['hostComponent = "www.apple.com"'])
  print(urlc.path) //% self.expect('frame variable -d run --  urlc', substrs=['portComponent = 0x', 'Int64(12345)'])
  print(urlc.query) //% self.expect('frame variable -d run --  urlc', substrs=['pathComponent = "/thisurl/isnotreal/itoldyou.php"'])
  print("break here last") //% self.expect('frame variable -d run --  urlc', substrs=['queryComponent = "page=fake"'])
  //% self.expect('expression -d run --  urlc', substrs=['urlString = "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake"', 'schemeComponent = "https"', 'userComponent = nil', 'fragmentComponent = nil'])
}

main()
