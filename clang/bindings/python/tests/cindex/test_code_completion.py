from clang.cindex import (
    AvailabilityKind,
    CompletionChunk,
    CompletionChunkKind,
    CompletionString,
    SPELLING_CACHE,
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
            "{'int', ResultType} | {'test1', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Aaa.",
            "{'void', ResultType} | {'test2', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Bbb.",
            "{'return', TypedText} | {';', SemiColon} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
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
            "{'int', ResultType} | {'test1', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Aaa.",
            "{'void', ResultType} | {'test2', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: Bbb.",
            "{'return', TypedText} | {';', SemiColon} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
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
            "{'const', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'volatile', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'operator', TypedText} || Priority: 40 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'P', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'Q', TypedText} || Priority: 50 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

        cr = tu.codeComplete("fake.cpp", 13, 5, unsaved_files=files)
        expected = [
            "{'P', TypedText} | {'::', Text} || Priority: 75 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'P &', ResultType} | {'operator=', TypedText} | {'(', LeftParen} | {'const P &', Placeholder} | {')', RightParen} || Priority: 79 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
            "{'int', ResultType} | {'member', TypedText} || Priority: 35 || Availability: AvailabilityKind.NOT_ACCESSIBLE || Brief comment: ",
            "{'void', ResultType} | {'~P', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 79 || Availability: AvailabilityKind.AVAILABLE || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

    def test_completion_chunk_kind_compatibility(self):
        value_to_old_str = {
            0: "Optional",
            1: "TypedText",
            2: "Text",
            3: "Placeholder",
            4: "Informative",
            5: "CurrentParameter",
            6: "LeftParen",
            7: "RightParen",
            8: "LeftBracket",
            9: "RightBracket",
            10: "LeftBrace",
            11: "RightBrace",
            12: "LeftAngle",
            13: "RightAngle",
            14: "Comma",
            15: "ResultType",
            16: "Colon",
            17: "SemiColon",
            18: "Equal",
            19: "HorizontalSpace",
            20: "VerticalSpace",
        }

        # Check that all new kinds correspond to an old kind
        for new_kind in CompletionChunkKind:
            old_str = value_to_old_str[new_kind.value]
            with warnings.catch_warnings(record=True) as log:
                self.assertEqual(old_str, str(new_kind))
                self.assertEqual(len(log), 1)
                self.assertIsInstance(log[0].message, DeprecationWarning)

        # Check that all old kinds correspond to a new kind
        for value, old_str in value_to_old_str.items():
            new_kind = CompletionChunkKind.from_id(value)
            with warnings.catch_warnings(record=True) as log:
                self.assertEqual(old_str, str(new_kind))
                self.assertEqual(len(log), 1)
                self.assertIsInstance(log[0].message, DeprecationWarning)

    def test_spelling_cache_missing_attribute(self):
        # Test that accessing missing attributes on SpellingCacheAlias raises
        # during the transitionary period
        with self.assertRaises(AttributeError, msg=SPELLING_CACHE.deprecation_message):
            SPELLING_CACHE.keys()

    def test_spelling_cache_alias(self):
        kind_keys = list(CompletionChunk.SPELLING_CACHE)
        self.assertEqual(len(kind_keys), 13)
        for kind_key in kind_keys:
            with warnings.catch_warnings(record=True) as log:
                self.assertEqual(
                    SPELLING_CACHE[kind_key.value],
                    CompletionChunk.SPELLING_CACHE[kind_key],
                )
                self.assertEqual(len(log), 1)
                self.assertIsInstance(log[0].message, DeprecationWarning)

    def test_spelling_cache_missing_attribute(self):
        # Test that accessing missing attributes on SpellingCacheAlias raises
        # during the transitionary period
        with self.assertRaises(AttributeError, msg=SPELLING_CACHE.deprecation_message):
            SPELLING_CACHE.keys()
