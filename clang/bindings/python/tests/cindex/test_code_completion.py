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


class TestCodeCompletion(unittest.TestCase):
    def check_completion_results(self, cr, expected):
        self.assertIsNotNone(cr)
        self.assertEqual(len(cr.diagnostics), 0)

        completions = [str(c) for c in cr.results]

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
            "{'int', ResultType} | {'test1', TypedText} || Priority: 50 || Availability: Available || Brief comment: Aaa.",
            "{'void', ResultType} | {'test2', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 50 || Availability: Available || Brief comment: Bbb.",
            "{'return', TypedText} | {';', SemiColon} || Priority: 40 || Availability: Available || Brief comment: ",
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
            "{'int', ResultType} | {'test1', TypedText} || Priority: 50 || Availability: Available || Brief comment: Aaa.",
            "{'void', ResultType} | {'test2', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 50 || Availability: Available || Brief comment: Bbb.",
            "{'return', TypedText} | {';', SemiColon} || Priority: 40 || Availability: Available || Brief comment: ",
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
            "{'const', TypedText} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'volatile', TypedText} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'operator', TypedText} || Priority: 40 || Availability: Available || Brief comment: ",
            "{'P', TypedText} || Priority: 50 || Availability: Available || Brief comment: ",
            "{'Q', TypedText} || Priority: 50 || Availability: Available || Brief comment: ",
        ]
        self.check_completion_results(cr, expected)

        cr = tu.codeComplete("fake.cpp", 13, 5, unsaved_files=files)
        expected = [
            "{'P', TypedText} | {'::', Text} || Priority: 75 || Availability: Available || Brief comment: ",
            "{'P &', ResultType} | {'operator=', TypedText} | {'(', LeftParen} | {'const P &', Placeholder} | {')', RightParen} || Priority: 79 || Availability: Available || Brief comment: ",
            "{'int', ResultType} | {'member', TypedText} || Priority: 35 || Availability: NotAccessible || Brief comment: ",
            "{'void', ResultType} | {'~P', TypedText} | {'(', LeftParen} | {')', RightParen} || Priority: 79 || Availability: Available || Brief comment: ",
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
            self.assertEqual(str(kind), string)

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
            self.assertEqual(old_str, str(new_kind))

        # Check that all old kinds correspond to a new kind
        for value, old_str in value_to_old_str.items():
            new_kind = CompletionChunkKind.from_id(value)
            self.assertEqual(old_str, str(new_kind))

    def test_spelling_cache_missing_attribute(self):
        # Test that accessing missing attributes on SpellingCacheAlias raises
        # during the transitionary period
        with self.assertRaises(AttributeError, msg=SPELLING_CACHE.deprecation_message):
            SPELLING_CACHE.keys()

    def test_spelling_cache_alias(self):
        kind_keys = list(CompletionChunk.SPELLING_CACHE)
        self.assertEqual(len(kind_keys), 13)
        for kind_key in kind_keys:
            self.assertEqual(
                SPELLING_CACHE[kind_key.value], CompletionChunk.SPELLING_CACHE[kind_key]
            )

    def test_spelling_cache_missing_attribute(self):
        # Test that accessing missing attributes on SpellingCacheAlias raises
        # during the transitionary period
        with self.assertRaises(AttributeError, msg=SPELLING_CACHE.deprecation_message):
            SPELLING_CACHE.keys()
