#!/usr/bin/env python3
R"""
TLDR:
The order for a single code snippet example is:

  \header{a.h}
  \endheader     <- zero or more header

  \code
    int a = 42;
  \endcode
  \compile_args{-std=c++,c23-or-later} <- optional, supports std ranges and
                                          whole languages

  \matcher{expr()} <- one or more matchers in succession
  \match{42}   <- one ore more matches in succession

  \matcher{varDecl()} <- new matcher resets the context, the above
                         \match will not count for this new
                         matcher(-group)
  \match{int a  = 42} <- only applies to the previous matcher (no the
                         previous case)

The above block can be repeated inside of a doxygen command for multiple
code examples.

Language Grammar:
  [] denotes an optional, and <> denotes user-input

  compile_args j:= \compile_args{[<compile_arg>;]<compile_arg>}
  matcher_tag_key ::= type
  match_tag_key ::= type || std || count
  matcher_tags ::= [matcher_tag_key=<value>;]matcher_tag_key=<value>
  match_tags ::= [match_tag_key=<value>;]match_tag_key=<value>
  matcher ::= \matcher{[matcher_tags$]<matcher>}
  matchers ::= [matcher] matcher
  match ::= \match{[match_tags$]<match>}
  matches ::= [match] match
  case ::= matchers matches
  cases ::= [case] case
  header-block ::= \header{<name>} <code> \endheader
  code-block ::= \code <code> \endcode
  testcase ::= code-block [compile_args] cases

The 'std' tag and '\compile_args' support specifying a specific
language version, a whole language and all of it's versions, and thresholds
(implies ranges). Multiple arguments are passed with a ',' seperator.
For a language and version to execute a tested matcher, it has to match
the specified '\compile_args' for the code, and the 'std' tag for the matcher.
Predicates for the 'std' compiler flag are used with disjunction between
languages (e.g. 'c || c++') and conjunction for all predicates specific
to each language (e.g. 'c++11-or-later && c++23-or-earlier').

Examples:
 - c                                    all available versions of C
 - c++11                                only C++11
 - c++11-or-later                       C++11 or later
 - c++11-or-earlier                     C++11 or earlier
 - c++11-or-later,c++23-or-earlier,c    all of C and C++ between 11 and
                                          23 (inclusive)
 - c++11-23,c                             same as above

Tags:

  Type:
  Match types are used to select where the string that is used to check if
  a node matches comes from.
  Available: code, name, typestr, typeofstr.
  The default is 'code'.

  Matcher types are used to mark matchers as submatchers with 'sub' or as
  deactivated using 'none'. Testing submatchers is not implemented.

  Count:
  Specifying a 'count=n' on a match will result in a test that requires that
  the specified match will be matched n times. Default is 1.

  Std:
  A match allows specifying if it matches only in specific language versions.
  This may be needed when the AST differs between language versions.
"""

import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from enum import Enum
from functools import reduce
from pathlib import Path
from sys import exit

statistics = defaultdict(int)

expected_failure_statistics = {
    "missing_tests": 10,
    "skipped_objc": 42,
    "none_type_matchers": 6,
}

found_issue = False


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="extract test cases from the documentation comments of AST"
        " matchers",
    )

    parser.add_argument(
        "--output-file",
        help="The path to the ASTMatchersDocTest.cpp file where the tests are"
        " written to.",
        type=Path,
    )

    parser.add_argument(
        "--input-file",
        help="The path to the ASTMatchers.h file",
        type=Path,
    )

    return parser.parse_args()


def matcher_to_node_type(matcher: str) -> str:
    """Converts the written top-level matcher to the node type that the matcher
    will bind.
    Contains special cases because not all matcher names can be simply converted
    to a type by spelling it with uppercase characters in the beginning.

    Args:
        matcher: The matcher string

    Returns:
        The type of the node that a match will bind to
    """
    if matcher.startswith("traverse"):
        comma_loc = matcher.find(",")
        return matcher_to_node_type(matcher[comma_loc + 1 :].strip())
    if matcher.startswith("cxxBoolLiteral"):
        return "CXXBoolLiteralExpr"
    if matcher.startswith("binaryOperation"):
        return "Expr"
    if matcher.startswith("invocation"):
        return "Expr"
    if matcher.startswith("alignOfExpr"):
        return "Expr"
    if matcher.startswith("sizeOfExpr"):
        return "Expr"
    if matcher.startswith("floatLiteral"):
        return "FloatingLiteral"
    if matcher.startswith("gnuNullExpr"):
        return "GNUNullExpr"
    if matcher.startswith("cxx"):
        return "CXX" + matcher[3 : matcher.find("(")]
    if matcher.startswith("omp"):
        return "OMP" + matcher[3 : matcher.find("(")]
    if matcher.startswith("cuda"):
        return "CUDA" + matcher[4 : matcher.find("(")]
    if matcher.startswith("objc"):
        return "ObjC" + matcher[4 : matcher.find("(")]
    return matcher[0:1].upper() + matcher[1 : matcher.find("(")]


def get_clang_config_constraint_expr(std: str) -> str:
    """Converts a single argument to 'std' into the corresponding check to be
    done in the tests.

    Args:
        std: A single argument to 'std' (e.g. 'c++11-or-later')

    Returns:
        An expression that checks is the test config enables what the 'std'
        argument specifies.
    """
    if std == "":
        return ""

    or_later_str = "-or-later"
    or_earlier_str = "-or-earlier"
    match = re.match(r"c(\d\d)-?(\d\d)?", std)
    if match:
        if len(match.groups()) == 3:
            return f"Conf.isCOrLater({match.group(1)}) && Conf.isOrEarlier({match.group(2)})"
        if std.endswith(or_later_str):
            return f"Conf.isCOrLater({match.group(1)})"
        if std.endswith(or_earlier_str):
            return f"Conf.isCOrEarlier({match.group(1)})"
        return f"Conf.Language == Lang_C{match.group(1)}"

    match = re.match(r"c\+\+(\d\d)-?(\d\d)?", std)
    if match:
        if len(match.groups()) == 3:
            return f"Conf.isCXXOrLater({match.group(1)}) && Conf.isCXXOrEarlier({match.group(2)})"
        if std.endswith(or_later_str):
            return f"Conf.isCXXOrLater({match.group(1)})"
        if std.endswith(or_earlier_str):
            return f"Conf.isCXXOrEarlier({match.group(1)})"
        return f"Conf.Language == Lang_CXX{match.group(1)}"

    if std == "c":
        return "Conf.isC()"

    if std == "c++":
        return "Conf.isCXX()"

    if std.startswith("-ObjC"):
        return ""

    return ""


class TestLanguage:
    """Wraps multiple args to 'std' for emitting the config check inside the
    tests.

    Attributes:
        raw: The arg to 'std' as written
        c: The config check expression for C
        cxx: The config check expression for C++
        objc: The config check expression for ObjC
    """

    def __init__(self, std: str) -> None:
        self.raw = std
        self.c = ""
        self.cxx = ""
        self.objc = ""
        if std.startswith("-ObjC"):
            self.objc = std
            return

        for standard_spec in std.split(","):
            expr = get_clang_config_constraint_expr(standard_spec)
            if standard_spec.startswith("c++"):
                if self.cxx != "":
                    self.cxx += " && "
                self.cxx += expr
            elif standard_spec.startswith("c"):
                if self.c != "":
                    self.c += " && "
                self.c += expr

    def has_value(self):
        return self.c != "" or self.cxx != "" or self.objc != ""

    def get_config_check_expr(self):
        if self.c != "" and self.cxx != "":
            return f"({self.c}) || ({self.cxx})"
        if self.c != "":
            return self.c
        if self.cxx != "":
            return self.cxx
        if self.objc != "":
            return self.objc
        return ""


class Tags:
    """Wrapper to parse and store the tags that can be passed to '\\match' and
    '\\matcher'.

    Attributes:
        map: The parsed tags by key ('str' -> 'str' map)
        opt_test_language: The 'std' tag is handled by TestLanguage
    """

    def __init__(self, tag_string: str) -> None:
        self.map = defaultdict(
            lambda: "",
            [split_first(tag, "=") for tag in tag_string.split(";")],
        )

        self.opt_test_language = None
        if "std" in self.map:
            self.opt_test_language = TestLanguage(self.map["std"])
            self.map.pop("std")


class Matcher:
    def __init__(self, matcher: str, tags: Tags) -> None:
        self.matcher = matcher
        self.tags = tags

    def is_sub_matcher(self) -> bool:
        return self.tags.map["type"] == "sub"

    def __format__(self, format_spec: str) -> str:
        return self.matcher.__format__(format_spec)

    def __str__(self) -> str:
        return f"{self.matcher}"

    def __repr__(self) -> str:
        return f"{self.matcher}"


class Match:
    def __init__(self, match_str: str, tags: Tags) -> None:
        self.match_str = match_str.replace("\n", " ")
        self.tags = tags

    def is_sub_match(self) -> bool:
        return self.tags.map["sub"] != ""

    def __format__(self, format_spec: str) -> str:
        return self.match_str.__format__(format_spec)

    def __str__(self) -> str:
        return f"{self.match_str}"

    def __repr__(self) -> str:
        return f"{self.match_str}"

    def get_as_cpp_raw_string(self) -> str:
        return f'R"cpp({self.match_str})cpp"'


def get_lang_spec_and_remove_from_list(args: list) -> str:
    for arg in args:
        if arg == "-ObjC":
            args.remove(arg)
            return arg
        if arg.startswith("-std="):
            args.remove(arg)
            return arg[5:]
    return ""


class CompileArgs:
    """Represents the '\\compile_args' command and its arguments.

    Attributes:
        lang_spec: The specified test language
        args: All other arguments
    """

    def __init__(self, args: list) -> None:
        self.lang_spec = TestLanguage(get_lang_spec_and_remove_from_list(args))
        self.args = args

        if any(("cuda" in arg for arg in self.args)) and not any(
            "-x" in arg for arg in self.args
        ):
            self.args.append("-xcuda")
            self.args.append("-nocudainc")
            self.args.append("-nocudalib")

    def is_cuda(self) -> bool:
        return any("cuda" in cmd for cmd in self.args)

    def is_objc(self) -> bool:
        return self.lang_spec.objc != ""


def get_any_valid_std_specified(lang_spec: str) -> str:
    """Get any argument to '-std' that satisfies the language specification
    in 'lang_spec'

    Args:
        lang_spec: An argument to 'std'

    Returns:
        Any valid argument to the '-std' compiler flag that satisfies
        'lang_spec'
    """
    if lang_spec == "":
        return "c++11"

    first_comma = lang_spec.find(",")
    if first_comma != -1:
        lang_spec = lang_spec[:first_comma]

    first_minus = lang_spec.find("-")
    if first_minus != -1:
        lang_spec = lang_spec[:first_minus]

    elif lang_spec == "c":
        lang_spec = "c11"
    elif lang_spec == "c++":
        lang_spec = "c++11"

    return lang_spec


def get_with_lang_spec(args: CompileArgs) -> list:
    """Construct compiler arguments from a CompileArgs instance

    Args:
        args: The arguments to '\\compile_args'

    Returns:
        A list of compiler arguments that satisfy what is specified in 'args'
    """
    if args.lang_spec.objc.startswith("-ObjC"):
        return [*args.args, args.lang_spec.objc]

    if args.lang_spec.has_value():
        return [*args.args, "-std=" + get_any_valid_std_specified(args.lang_spec.raw)]

    return args.args


cuda_header: str = """
    typedef unsigned long long size_t;
    #define __constant__ __attribute__((constant))
    #define __device__ __attribute__((device))
    #define __global__ __attribute__((global))
    #define __host__ __attribute__((host))
    #define __shared__ __attribute__((shared))
    struct dim3 {
    unsigned x, y, z;
    __host__ __device__ dim3(unsigned x, unsigned y = 1, unsigned z = 1)
        : x(x), y(y), z(z) {}
    };
    typedef struct cudaStream *cudaStream_t;
    int cudaConfigureCall(dim3 gridSize, dim3 blockSize,
                        size_t sharedSize = 0,
                        cudaStream_t stream = 0);
    extern "C" unsigned __cudaPushCallConfiguration(
      dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream =
    0);
"""


class MatchType(Enum):
    """Available types of a match, specified by using the lowercase version in
    doxygen: '\\match{type=typestr}'.
    Specifies what string from the matched node should be used to check for a
    match.

    Attributes:
        Invalid: Specified an invalid match type
        Code: Check if a matched node matches by using it's text
              (SourceRange -> Lexer)
        Name: Use the name of the matched node to check if it matches
        TypeStr: Use the string representation of the matched type to check if
                 it matches
    """

    Invalid = 0
    Code = 1
    Name = 2
    TypeStr = 3


def get_match_type(match: Match) -> MatchType:
    match_type = match.tags.map["type"]
    if match_type == "name":
        return MatchType.Name
    if match_type == "" or match_type == "code":
        return MatchType.Code
    if match_type == "typestr":
        return MatchType.TypeStr
    print(f"match {match} has an invalid match type: {match_type}")
    statistics["match_type_invalid"] += 1
    return MatchType.Invalid


def get_match_count(match: Match) -> str:
    count = match.tags.map["count"]
    if count == "":
        return "1"
    return count


def construct_match(match: Match) -> str:
    result = ""
    if match.tags.opt_test_language:
        result += f"\tif ({match.tags.opt_test_language.get_config_check_expr()})\n\t"

    result += f"\tMatches.emplace_back(MatchKind::{get_match_type(match).name}, {match.get_as_cpp_raw_string()}, {get_match_count(match)});"
    return result


class TestCase:
    """Represents a single code example and its tests. Tests are added during
    parsing and finally emitted as C++ test code.

    Attributes:
        input_file: The file path where the matcher was specified
        line: The line that the test case starts at
        cases: The collected cases for this example code
        code: The example code
        headers: Headers, if any
        compile_args: Compile arguments, if any
    """

    def __init__(self, input_file: Path, line: int):
        self.input_file = input_file
        self.line = line
        self.cases: list[tuple[list[Matcher], list[Match]]] = []
        self.code: str = ""
        self.headers: list[tuple[str, str]] = []
        self.compile_args = None

    def get_last_case(self) -> tuple:
        return self.cases[-1]

    def add_case(self):
        self.cases.append(([], []))

    def add_matcher(self, matcher: Matcher):
        self.get_last_case()[0].append(matcher)

    def add_match(self, match: Match):
        self.get_last_case()[1].append(match)

    def has_non_lang_spec_compile_args(self) -> bool:
        return self.compile_args is not None and len(self.compile_args.args) != 0

    def has_headers(self) -> bool:
        return len(self.headers) != 0

    def get_compile_commands(self) -> str:
        if self.compile_args is None:
            return "{}"
        return "{ " + ",".join([f'"{arg}"' for arg in self.compile_args.args]) + " }"

    def diag(self, message: str):
        print(f"{self.input_file}:{self.line + 1}: {message}")

    def has_match_with_std_constraint(self) -> bool:
        return any(
            (
                any(match.tags.opt_test_language for match in case[1])
                for case in self.cases
            ),
        )

    def get_formated_headers(self) -> str:
        return "\n\t".join(
            f'std::pair{{"{header[0]}", \n\tR"cpp({header[1]})cpp"}},'
            for header in self.headers
        )

    def build_test_case(self):
        self.code = self.code.strip("\n")
        has_cuda = self.compile_args and self.compile_args.is_cuda()
        if has_cuda:
            self.headers.append(("cuda.h", cuda_header))

        res = ""
        if has_cuda:
            res += "#if LLVM_HAS_NVPTX_TARGET\n"

        res += f"""TEST_P(ASTMatchersDocTest, docs_{self.line + 1}) {{
    const StringRef Code = R"cpp(\n"""
        if has_cuda:
            res += '\t#include "cuda.h"\n'
        res += f'{self.code})cpp";\n'

        if self.has_headers():
            res += f"\tconst FileContentMappings VirtualMappedFiles = {{{self.get_formated_headers()}}};"

        if self.compile_args and self.compile_args.lang_spec.objc != "":
            statistics["skipped_objc"] += 1
            return ""

        statistics["code_snippets"] += 1

        statistics["matches"] += reduce(
            lambda a, b: a + b,
            (len(v[1]) for v in self.cases),
        )

        statistics["matchers"] += reduce(
            lambda a, b: a + b,
            (len(v[0]) for v in self.cases),
        )

        if self.compile_args and self.compile_args.lang_spec.has_value():
            res += f"""
    const TestClangConfig& Conf = GetParam();
    const bool ConfigEnablesCheckingCode = {self.compile_args.lang_spec.get_config_check_expr()};
    if (!ConfigEnablesCheckingCode) return;\n"""

        elif self.has_match_with_std_constraint():
            res += "\n\tconst TestClangConfig& Conf = GetParam();\n"

        # Don't want to emit a test without any actual tested matchers.
        # Would result in warnings about unused variables (Code) during the
        # build.
        case_has_test = False

        for matchers, matches in self.cases:
            if len(matchers) == 0:
                self.diag("test cases have no matchers")
                continue

            for matcher in matchers:
                if matcher.matcher.startswith("objc"):
                    statistics["skipped_objc"] += 1
                    continue
                # FIXME: add support for testing submatchers
                if matcher.tags.map["type"] == "sub":
                    continue
                if matcher.tags.map["type"] == "none":
                    statistics["none_type_matchers"] += 1
                    continue
                matcher_type = matcher_to_node_type(matcher.matcher)

                statistics["tested_matchers"] += 1
                case_has_test = True

                verifier_type = f"VerifyBoundNodeMatch<{matcher_type}>"

                code_adding_matches = "\n".join(
                    construct_match(match)
                    for match in matches
                    if match.tags.map["sub"] == ""
                )

                match_function = (
                    "matches" if len(code_adding_matches) != 0 else "notMatches"
                )

                res += f"""
    {{
    SCOPED_TRACE("Test failure from docs_{self.line + 1} originates here.");
    using Verifier = {verifier_type};

    std::vector<Verifier::Match> Matches;
{code_adding_matches}

    EXPECT_TRUE({match_function}(
        Code,
        {matcher.matcher}.bind("match"),
        std::make_unique<Verifier>("match", Matches)"""

                if self.has_headers():
                    res += f",\n\t\t{self.get_compile_commands()}"
                    res += ",\n\t\tVirtualMappedFiles"
                elif self.has_non_lang_spec_compile_args():
                    res += f",\n\t\t{self.get_compile_commands()}"

                res += "));\n\t}\n"

        if not case_has_test:
            return ""

        res += "}"
        if has_cuda:
            res += "\n#endif\n"
        return res


class ParsingContext(Enum):
    NoneCtx = 0
    Code = 1
    Header = 2
    CompileArgs = 3
    Matcher = 4
    Match = 5
    NoMatch = 6


def split_first(data: str, delim: str) -> tuple:
    pos = data.find(delim)
    if pos == -1:
        return (data, "")

    return (data[0:pos], data[pos + 1 :])


def find_matching_closing_rbrace(
    data: str,
    start_pos: int,
    braces_to_be_matched: int,
) -> int:
    next_lbrace = data.find("{", start_pos)
    next_rbrace = data.find("}", start_pos)
    if next_lbrace != -1:
        if next_lbrace < next_rbrace:
            return find_matching_closing_rbrace(
                data,
                next_lbrace + 1,
                braces_to_be_matched + 1,
            )
        if braces_to_be_matched == 0:
            return next_rbrace
        return find_matching_closing_rbrace(
            data,
            next_rbrace + 1,
            braces_to_be_matched - 1,
        )

    if braces_to_be_matched > 0:
        return find_matching_closing_rbrace(
            data,
            next_rbrace + 1,
            braces_to_be_matched - 1,
        )

    return next_rbrace


class CommentedMatcher:
    """Represents a matcher and it's corresponding doxygen comment.

    Attributes:
        comment: The doxygen comment
        matcher: The actual matcher code/implementation
    """

    def __init__(
        self,
        comment: list,
        matcher: list,
    ):
        self.comment = comment
        self.matcher = matcher


class BlockParser:
    """The parser that parses the test cases from a doxygen block.

    Attributes:
        input_file: The file where the matchers are located
        cases: All TestCases
        current_line: The line that the parser is currently on
        data: The lines in the comment
    """

    def __init__(self, data: CommentedMatcher, input_file: Path):
        self.input_file = input_file
        self.cases = list()
        self.current_line = data.comment[0][0]
        self.data = "\n".join([line[1] for line in data.comment])
        self.diagnostics = list()

    def advance(self) -> ParsingContext:
        """Find the next doxygen command to be parsed and return the
        ParsingContext it represents.

        Returns:
            The ParsingContext of the location the parser arived at.
        """
        begin_tags = {
            "\\code": ParsingContext.Code,
            "\\header{": ParsingContext.Header,
            "\\compile_args{": ParsingContext.CompileArgs,
            "\\matcher{": ParsingContext.Matcher,
            "\\match{": ParsingContext.Match,
            "\\nomatch{": ParsingContext.NoMatch,
        }

        matches = list()

        for tag, ctx in begin_tags.items():
            match = self.data.find(tag)
            if match == -1:
                continue
            matches.append((match, ctx, tag))

        if len(matches) == 0:
            return ParsingContext.NoneCtx

        matches.sort()

        loc, ctx, tag = matches[0]
        loc = loc + len(tag) - 1
        self.consume_until(loc)
        return ctx

    def add_case(self, line: int):
        self.cases.append(TestCase(self.input_file, line))

    def delete_last_test(self):
        self.cases.pop()

    def get_last_case(self) -> TestCase:
        return self.cases[-1]

    def add_matcher(self, matcher: Matcher):
        if matcher.is_sub_matcher():
            return
        self.get_last_case().add_matcher(matcher)

    def add_match(self, match: Match):
        if match.is_sub_match():
            return
        self.get_last_case().add_match(match)

    def consume_until(self, pos: int):
        self.current_line += self.data.count("\n", 0, pos)
        self.data = self.data[pos + 1 :]

    def visit_code_block(self):
        code_end = "\\endcode"
        endcode_loc = self.data.find(code_end)
        end = endcode_loc + len(code_end)

        self.get_last_case().code = self.data[:endcode_loc].replace("\n", "\n\t")

        self.consume_until(end)

    def visit_header(self):
        header_end = "\\endheader"
        endheader_loc = self.data.find(header_end)
        end = endheader_loc + len(header_end)

        vrbrace_loc = find_matching_closing_rbrace(self.data, 0, 0)
        header_name = self.data[0:vrbrace_loc]

        self.get_last_case().headers.append(
            (
                header_name,
                self.data[vrbrace_loc + 1 : endheader_loc].replace("\n", "\n\t"),
            ),
        )

        self.consume_until(end)

    def visit_compile_args(self):
        end = find_matching_closing_rbrace(self.data, 0, 0)

        args = self.data[:end]
        self.consume_until(end)

        self.get_last_case().compile_args = CompileArgs(
            [split_arg for arg in args.split(";") for split_arg in arg.split(" ")],
        )

    def build_matcher(self):
        end = find_matching_closing_rbrace(self.data, 0, 0)
        tag_separator = self.data.find("$")
        tags = Tags("")
        if tag_separator != -1 and tag_separator < end:
            tags = Tags(self.data[:tag_separator])
            self.consume_until(tag_separator)

            # find the '}' again, self.data shifted
            end = find_matching_closing_rbrace(self.data, 0, 0)

        matcher_str = self.data[:end]
        if matcher_str.count("(") != matcher_str.count(")"):
            self.diag(
                f"The matcher {matcher_str} has an unbalanced number of parentheses"
            )
            self.consume_until(end)
            return None

        matcher = Matcher(self.data[:end], tags)
        self.consume_until(end)
        return matcher

    def visit_match(self):
        end = find_matching_closing_rbrace(self.data, 0, 0)
        tag_separator = self.data.find("$")
        tags = Tags("")
        if tag_separator != -1 and tag_separator < end:
            tags = Tags(self.data[:tag_separator])
            self.consume_until(tag_separator)

            # find the '}' again, self.data shifted
            end = find_matching_closing_rbrace(self.data, 0, 0)
        self.add_match(Match(self.data[:end], tags))
        self.consume_until(end)

    def diag(self, msg: str):
        # Save diagnostics instead of emitting them to remove the noise of diagnosed issues,
        # when the issues are all expected.
        self.diagnostics.append(f"{self.input_file}:{self.current_line + 1}: {msg}")

    def run(self):
        ctx = self.advance()

        if ctx == ParsingContext.NoneCtx:
            self.diag("matcher is missing an example")
            statistics["missing_tests"] += 1
            return

        while ctx != ParsingContext.NoneCtx:
            self.add_case(self.current_line)
            while ctx == ParsingContext.Header:
                self.visit_header()
                ctx = self.advance()

            if ctx != ParsingContext.Code:
                self.diag(f"expected {ParsingContext.Code}, not {ctx}")
                statistics["missing_tests"] += 1
                self.delete_last_test()
                return

            self.visit_code_block()
            ctx = self.advance()

            if ctx == ParsingContext.CompileArgs:
                self.visit_compile_args()
                compile_args = self.get_last_case().compile_args
                ctx = self.advance()
                if compile_args and compile_args.is_objc():
                    self.delete_last_test()
                    statistics["skipped_objc"] += 1

                    while ctx != ParsingContext.Code and ctx != ParsingContext.NoneCtx:
                        ctx = self.advance()

                    continue

            if ctx != ParsingContext.Matcher:
                if ctx == ParsingContext.NoneCtx:
                    self.diag(
                        "this code example is missing an example matcher and matches",
                    )
                else:
                    self.diag(
                        f"expected {ParsingContext.Matcher} after {ParsingContext.Code}, not {ctx}",
                    )

                statistics["missing_tests"] += 1
                self.delete_last_test()
                return

            while ctx == ParsingContext.Matcher:
                matcher = self.build_matcher()
                if matcher and not matcher.is_sub_matcher():
                    self.get_last_case().add_case()

                if matcher:
                    self.add_matcher(matcher)

                ctx = self.advance()
                while ctx == ParsingContext.Matcher:
                    matcher = self.build_matcher()
                    if matcher:
                        self.add_matcher(matcher)
                    ctx = self.advance()

                if ctx != ParsingContext.Match and ctx != ParsingContext.NoMatch:
                    if ctx == ParsingContext.NoneCtx:
                        self.diag(
                            "this matcher does not specify what it should or shouldn't match",
                        )
                    else:
                        self.diag(
                            f"expected {ParsingContext.Match} or {ParsingContext.NoMatch} after {ParsingContext.Matcher}, not {ctx}",
                        )

                    statistics["matcher_groups_without_matches"] += 1
                    break

                while ctx == ParsingContext.Match or ctx == ParsingContext.NoMatch:
                    if ctx == ParsingContext.Match:
                        self.visit_match()
                    ctx = self.advance()


def parse_block(data: CommentedMatcher, input_file: Path) -> BlockParser:
    parser = BlockParser(data, input_file)
    parser.run()
    return parser


def parse(data: list, input_file: Path) -> tuple:
    result: tuple = ([], [])

    for parsed_block in [parse_block(block, input_file) for block in data]:
        result[0].extend(parsed_block.cases)
        result[1].extend(parsed_block.diagnostics)

    return result


def group_doc_comment_and_followed_code(
    enumerated_lines: list,
) -> list:
    result: list = []

    start_new_group_on_comment = True
    for line_nr, line in enumerate(enumerated_lines, start=1):
        if line.startswith("///"):
            if start_new_group_on_comment:
                result.append(CommentedMatcher([], []))
            start_new_group_on_comment = False
            if len(result) != 0:
                result[-1].comment.append((line_nr, line[4:].rstrip()))
        else:
            start_new_group_on_comment = True
            if len(result) != 0:
                result[-1].matcher.append((line_nr, line))

    return result


test_file_begin = """
// unittests/ASTMatchers/ASTMatchersNarrowingTest.cpp - AST matcher unit tests//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTMatchersTest.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Testing/TestClangConfig.h"
#include "gtest/gtest.h"
#include <string>

namespace clang {
namespace ast_matchers {
"""

test_file_end = """
static std::vector<TestClangConfig> allTestClangConfigs() {
  std::vector<TestClangConfig> all_configs;
  for (TestLanguage lang : {
#define TESTLANGUAGE(lang, version, std_flag, index) Lang_##lang##version,
#include "clang/Testing/TestLanguage.def"
       }) {
    TestClangConfig config;
    config.Language = lang;

    // Use an unknown-unknown triple so we don't instantiate the full system
    // toolchain.  On Linux, instantiating the toolchain involves stat'ing
    // large portions of /usr/lib, and this slows down not only this test, but
    // all other tests, via contention in the kernel.
    //
    // FIXME: This is a hack to work around the fact that there's no way to do
    // the equivalent of runToolOnCodeWithArgs without instantiating a full
    // Driver.  We should consider having a function, at least for tests, that
    // invokes cc1.
    config.Target = "i386-unknown-unknown";
    all_configs.push_back(config);

    // Windows target is interesting to test because it enables
    // `-fdelayed-template-parsing`.
    config.Target = "x86_64-pc-win32-msvc";
    all_configs.push_back(config);
  }
  return all_configs;
}

INSTANTIATE_TEST_SUITE_P(
    ASTMatchersTests, ASTMatchersDocTest,
    testing::ValuesIn(allTestClangConfigs()),
    [](const testing::TestParamInfo<TestClangConfig> &Info) {
      return Info.param.toShortString();
    });
} // namespace ast_matchers
} // namespace clang
"""


def main():
    args = parse_arguments()

    assert args.input_file.exists()

    comment_blocks = group_doc_comment_and_followed_code(
        args.input_file.read_text().split("\n"),
    )

    statistics["doxygen_blocks"] = len(comment_blocks)
    results = parse(comment_blocks, args.input_file)

    res: str = (
        "\n\n".join(
            filter(
                lambda case: len(case) != 0,
                [r.build_test_case() for r in results[0]],
            ),
        )
        + "\n"
    )

    args.output_file.write_text(test_file_begin + res + test_file_end)

    global found_issue
    for key, value in expected_failure_statistics.items():
        if value != statistics[key]:
            print(
                "Mismatch between expected and actual failure statistic: "
                f"{key}: expected: {value}, actual: {statistics[key]}. "
                "Please fix the issue or adjust the expected_failure_statistics"
                " value if appropriate."
            )
            found_issue = True

    if found_issue:
        for diag in results[1]:
            print(diag)

        print("Statistics:")
        for key, value in statistics.items():
            print(f"\t{key: <30}: {value: >5}")

        exit(1)


# FIXME: add support in the html gen script to differentiate between overloads examples
# FIXME: should verify that all polymorphic ast nodes have a test


if __name__ == "__main__":
    main()
