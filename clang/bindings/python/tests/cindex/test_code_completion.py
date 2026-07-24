from clang.cindex import (
    AvailabilityKind,
    CompletionChunk,
    CompletionChunkKind,
    CompletionString,
    TranslationUnit,
)

import unittest
from pathlib import Path
import warnings


class TestCodeCompletion(unittest.TestCase):
    def check_completion_results(self, cr, expected):
        self.assertIsNotNone(cr)
        self.assertEqual(len(cr.diagnostics), 0)

        with warnings.catch_warnings(record=True) as log:
            completions = [str(c) for c in cr]
            self.assertEqual(len(log), 1)
            for warning in log:
                self.assertIsInstance(warning.message, DeprecationWarning)

        for c in expected:
            self.assertIn(c, completions)

        with warnings.catch_warnings(record=True) as log:
            completions_deprecated = [str(c) for c in cr.results]
            self.assertEqual(len(log), 2)
            for warning in log:
                self.assertIsInstance(warning.message, DeprecationWarning)

        for c in expected:
            self.assertIn(c, completions_deprecated)

    def test_code_complete(self):
        files = [
            (
                "fake.c",
                """
/// Aaa.
int test1;

/// Bbb.
void test2(void);

void f() {

}
""",
            )
        ]

        tu = TranslationUnit.from_source(
            "fake.c",
            ["-std=c99"],
            unsaved_files=files,
            options=TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION,
        )

        cr = tu.codeComplete(
            "fake.c", 9, 1, unsaved_files=files, include_brief_comments=True
        )

        expected = [
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'test1', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: Aaa.",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'test2', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 50 || Availability: Available || Brief comment: Bbb.",
            "{'return', CompletionChunkKind.TYPED_TEXT} | {';', CompletionChunkKind.SEMI_COLON} || Priority: 40 || Availability: Available || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

    def test_code_complete_pathlike(self):
        files = [
            (
                Path("fake.c"),
                """
/// Aaa.
int test1;

/// Bbb.
void test2(void);

void f() {

}
""",
            )
        ]

        tu = TranslationUnit.from_source(
            Path("fake.c"),
            ["-std=c99"],
            unsaved_files=files,
            options=TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION,
        )

        cr = tu.codeComplete(
            Path("fake.c"),
            9,
            1,
            unsaved_files=files,
            include_brief_comments=True,
        )

        expected = [
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'test1', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: Aaa.",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'test2', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 50 || Availability: Available || Brief comment: Bbb.",
            "{'return', CompletionChunkKind.TYPED_TEXT} | {';', CompletionChunkKind.SEMI_COLON} || Priority: 40 || Availability: Available || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

    def test_code_complete_availability(self):
        files = [
            (
                "fake.cpp",
                """
class P {
protected:
  int member;
};

class Q : public P {
public:
  using P::member;
};

void f(P x, Q y) {
  x.; // member is inaccessible
  y.; // member is accessible
}
""",
            )
        ]

        tu = TranslationUnit.from_source(
            "fake.cpp", ["-std=c++98"], unsaved_files=files
        )

        cr = tu.codeComplete("fake.cpp", 12, 5, unsaved_files=files)

        expected = [
            "{'const', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'volatile', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'operator', CompletionChunkKind.TYPED_TEXT} || Priority: 40 || Availability: Available || Brief comment: ",
            "{'P', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'Q', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: Available || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

        cr = tu.codeComplete("fake.cpp", 13, 5, unsaved_files=files)
        expected = [
            "{'P', CompletionChunkKind.TYPED_TEXT} | {'::', CompletionChunkKind.TEXT} || Priority: 75 || Availability: Available || Brief comment: ",
            "{'P &', CompletionChunkKind.RESULT_TYPE} | {'operator=', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {'const P &', CompletionChunkKind.PLACEHOLDER} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 79 || Availability: Available || Brief comment: ",
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'member', CompletionChunkKind.TYPED_TEXT} || Priority: 35 || Availability: NotAccessible || Brief comment: ",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'~P', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 79 || Availability: Available || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

    def test_availability_kind_compat(self):
        numKinds = len(CompletionString.AvailabilityKindCompat)

        # Compare with regular kind
        for compatKind in CompletionString.AvailabilityKindCompat:
            commonKind = AvailabilityKind.from_id(compatKind.value)
            nextKindId = (compatKind.value + 1) % numKinds
            commonKindUnequal = AvailabilityKind.from_id(nextKindId)
            self.assertEqual(commonKind, compatKind)
            self.assertEqual(compatKind, commonKind)
            self.assertNotEqual(commonKindUnequal, compatKind)
            self.assertNotEqual(compatKind, commonKindUnequal)

        # Compare two compat kinds
        for compatKind in CompletionString.AvailabilityKindCompat:
            compatKind2 = CompletionString.AvailabilityKindCompat.from_id(
                compatKind.value
            )
            nextKindId = (compatKind.value + 1) % numKinds
            compatKind2Unequal = CompletionString.AvailabilityKindCompat.from_id(
                nextKindId
            )
            self.assertEqual(compatKind, compatKind2)
            self.assertEqual(compatKind2, compatKind)
            self.assertNotEqual(compatKind2Unequal, compatKind)
            self.assertNotEqual(compatKind, compatKind2Unequal)

    def test_compat_str(self):
        kindStringMap = {
            0: "Available",
            1: "Deprecated",
            2: "NotAvailable",
            3: "NotAccessible",
        }
        for id, string in kindStringMap.items():
            kind = CompletionString.AvailabilityKindCompat.from_id(id)
            with warnings.catch_warnings(record=True) as log:
                self.assertEqual(str(kind), string)
                self.assertEqual(len(log), 1)
                self.assertIsInstance(log[0].message, DeprecationWarning)
