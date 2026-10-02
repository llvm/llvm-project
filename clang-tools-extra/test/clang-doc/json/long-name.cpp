// UNSUPPORTED: system-windows
// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %S/../Inputs/long-name.cpp
// RUN: ls %t/json/GlobalNamespace | FileCheck %s -check-prefix=CHECK-JSON

// CHECK-JSON: ThisStructHasANameThatResultsInAMangledNameThatIsExactly250CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd12.json
// CHECK-JSON: _ZTV244ThisStructHasANameThatResultsInAMangledNameThatIsExactly251CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheL29DE8558215A13A506661C0E01E50AA3E5C9C7FA.json
