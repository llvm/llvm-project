// PlaygroundsRuntime.swift
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

// If you're modifying this file to add or modify function signatures, you
// should be notifying the maintainer of PlaygroundLogger and also the
// maintainer of lib/Sema/PlaygroundTransform.cpp.

var PlaygroundOutput = ""

struct SourceRange {
  let sl : Int
  let el : Int
  let sc : Int
  let ec : Int
  var text : String {
    return "[\(sl):\(sc)-\(el):\(ec)]"
  }
}

class LogRecord {
  let text : String

  init(api : String, object : Any, name : String, id : Int, range : SourceRange) {
    var object_description : String = ""
    print(object, terminator: "", to: &object_description)
    text = range.text + " " + api + "[" + name + "='" + object_description + "']"
  }
  init(api : String, object : Any, name : String, range : SourceRange) {
    var object_description : String = ""
    print(object, terminator: "", to: &object_description)
    text = range.text + " " + api + "[" + name + "='" + object_description + "']"
  }
  init(api : String, object: Any, range : SourceRange) {
    var object_description : String = ""
    print(object, terminator: "", to: &object_description)
    text = range.text + " " + api + "['" + object_description + "']"
  }
  init(api: String, range : SourceRange) {
    text = range.text + " " + api
  }
}

@_silgen_name ("playground_logger_initialize") public func builtin_initialize() {
}

@_silgen_name ("playground_log_hidden") public func builtin_log_with_id<T>(_ object : T, _ name : String, _ id : Int, _ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject? {
  return LogRecord(api:"__builtin_log", object:object, name:name, id:id, range : SourceRange(sl:sl, el:el, sc:sc, ec:ec))
}

@_silgen_name ("playground_log_scope_entry") public func builtin_log_scope_entry(_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject? {
  return LogRecord(api:"__builtin_log_scope_entry", range : SourceRange(sl:sl, el:el, sc:sc, ec:ec))
}

@_silgen_name ("playground_log_scope_exit") public func builtin_log_scope_exit(_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject? {
  return LogRecord(api:"__builtin_log_scope_exit", range : SourceRange(sl:sl, el:el, sc:sc, ec:ec))
}

@_silgen_name ("playground_log_postprint") public func builtin_postPrint(_ sl : Int, _ el : Int, _ sc : Int, _ ec: Int) -> AnyObject? {
  return LogRecord(api:"__builtin_postPrint", range : SourceRange(sl:sl, el:el, sc:sc, ec:ec))
}

@_silgen_name ("DVTSendPlaygroundLogData") public func builtin_send_data(_ object:AnyObject?) {
  print((object as! LogRecord).text)
  PlaygroundOutput.append((object as! LogRecord).text)
  PlaygroundOutput.append("\n")
}

@_silgen_name ("GetOutput") public func get_output() -> String {
  return PlaygroundOutput
}

