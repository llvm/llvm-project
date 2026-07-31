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

        completions = [str(c) for c in cr]
        for c in expected:
            self.assertIn(c, completions)

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
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'test1', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Aaa.",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'test2', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Bbb.",
            "{'return', CompletionChunkKind.TYPED_TEXT} | {';', CompletionChunkKind.SEMI_COLON} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
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
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'test1', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Aaa.",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'test2', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Bbb.",
            "{'return', CompletionChunkKind.TYPED_TEXT} | {';', CompletionChunkKind.SEMI_COLON} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
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
            "{'const', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'volatile', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'operator', CompletionChunkKind.TYPED_TEXT} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'P', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'Q', CompletionChunkKind.TYPED_TEXT} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

        cr = tu.codeComplete("fake.cpp", 13, 5, unsaved_files=files)
        expected = [
            "{'P', CompletionChunkKind.TYPED_TEXT} | {'::', CompletionChunkKind.TEXT} || Priority: 75 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'P &', CompletionChunkKind.RESULT_TYPE} | {'operator=', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {'const P &', CompletionChunkKind.PLACEHOLDER} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 79 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'int', CompletionChunkKind.RESULT_TYPE} | {'member', CompletionChunkKind.TYPED_TEXT} || Priority: 35 || AvailabilityKind.NOT_ACCESSIBLE || Brief comment: ",
            "{'void', CompletionChunkKind.RESULT_TYPE} | {'~P', CompletionChunkKind.TYPED_TEXT} | {'(', CompletionChunkKind.LEFT_PAREN} | {')', CompletionChunkKind.RIGHT_PAREN} || Priority: 79 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
        ]
