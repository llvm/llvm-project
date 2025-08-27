// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=mustache --executor=standalone %s
// RUN: ls %t/json | FileCheck %s -check-prefix=CHECK-JSON
// RUN: ls %t/html | FileCheck %s -check-prefix=CHECK-HTML

struct ThisStructHasANameThatResultsInAMangledNameThatIsExactly250CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd12 {};

// This name is 1 character over the limit, so it will be serialized as a USR.
struct ThisStructHasANameThatResultsInAMangledNameThatIsExactly251CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd123 {};

// CHECK-JSON: ThisStructHasANameThatResultsInAMangledNameThatIsExactly250CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd12.json
// CHECK-JSON: {{[0-9A-F]*}}.json
// CHECK-HTML: ThisStructHasANameThatResultsInAMangledNameThatIsExactly250CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd12.html
// CHECK-HTML: {{[0-9A-F]*}}.html
