// UNSUPPORTED: system-windows
// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=html --executor=standalone %S/../Inputs/long-name.cpp
// RUN: ls %t/html/GlobalNamespace | FileCheck %s -check-prefix=CHECK-HTML

// CHECK-HTML: ThisStructHasANameThatResultsInAMangledNameThatIsExactly250CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheLengthIsTooLongThenClangDocWillCrashAnd12.html
// CHECK-HTML: _ZTV244ThisStructHasANameThatResultsInAMangledNameThatIsExactly251CharactersLongThatIsSupposedToTestTheFilenameLengthLimitsWithinClangDocInOrdertoSeeifclangdocwillcrashornotdependingonthelengthofthestructIfTheL29DE8558215A13A506661C0E01E50AA3E5C9C7FA.html
