// main.cpp
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

#import "library.h"

int main() {
  RuntimeErrorDetails::Note notes[] = {
      { .description = "note 1" },
      { .description = "note 2" },
      { .description = "note 3" },
  };
  RuntimeErrorDetails::FixIt fixits[] = {
    { .filename = "filename1", .startLine = 42, .startColumn = 1,
      .endLine = 43, .endColumn = 2, .replacementText = "replacement1" },
    { .filename = "filename2", .startLine = 44, .startColumn = 3,
      .endLine = 45, .endColumn = 4, .replacementText = "replacement2" }
  };
  RuntimeErrorDetails details = {
    .version = 2,
    .errorType = "my-error",
    .notes = &notes[0],
    .numNotes = 3,
    .fixIts = &fixits[0],
    .numFixIts = 2
  };
  _swift_runtime_on_report(RuntimeErrorFlagNone, "custom error message", &details);
}
