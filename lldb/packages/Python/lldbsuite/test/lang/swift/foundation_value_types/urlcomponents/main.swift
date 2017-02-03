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

func main() {
  var urlc = URLComponents(string: "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake")!
  print(urlc.scheme) //% self.expect('frame variable -d run -- urlc', substrs=['urlString = "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake"'])
  print(urlc.host) //% self.expect('frame variable -d run --  urlc', substrs=['scheme = "https"'])
  print(urlc.port) //% self.expect('frame variable -d run --  urlc', substrs=['host = "www.apple.com"'])
  print(urlc.path) //% self.expect('frame variable -d run --  urlc', substrs=['port = 0x', 'Int64(12345)'])
  print(urlc.query) //% self.expect('frame variable -d run --  urlc', substrs=['path = "/thisurl/isnotreal/itoldyou.php"'])
  print("break here last") //% self.expect('frame variable -d run --  urlc', substrs=['query = "page=fake"'])
  //% self.expect('expression -d run --  urlc', substrs=['urlString = "https://www.apple.com:12345/thisurl/isnotreal/itoldyou.php?page=fake"', 'scheme = "https"', 'user = nil', 'fragment = nil'])
}

main()
