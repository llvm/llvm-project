#!/usr/bin/env python3
# A tool to automatically generate documentation for the config options of the
# clang static analyzer by reading `AnalyzerOptions.def`.

import argparse
from collections import namedtuple
from enum import Enum, auto
import re
import sys
import textwrap


# The following code implements a trivial parser for the narrow subset of C++
# which is used in AnalyzerOptions.def. This supports the following features:
# - ignores preprocessor directives, even if they are continued with \ at EOL
# - ignores comments: both /* ... */ and // ...
# - parses string literals (even if they contain \" escapes)
# - concatenates adjacent string literals
# - parses numbers even if they contain ' as a thousands separator
# - recognizes MACRO(arg1, arg2, ..., argN) calls


class TT(Enum):
    "Token type enum."
    number = auto()
    ident = auto()
    string = auto()
    punct = auto()


TOKENS = [
    (re.compile(r"-?[0-9']+"), TT.number),
    (re.compile(r"\w+"), TT.ident),
    (re.compile(r'"([^\\"]|\\.)*"'), TT.string),
    (re.compile(r"[(),]"), TT.punct),
    (re.compile(r"/\*((?!\*/).)*\*/", re.S), None),  # C-style comment
    (re.compile(r"//.*\n"), None),  # C++ style oneline comment
    (re.compile(r"#.*(\\\n.*)*(?<!\\)\n"), None),  # preprocessor directive
    (re.compile(r"\s+"), None),  # whitespace
]

Token = namedtuple("Token", "kind code")


class ErrorHandler:
    def __init__(self):
        self.seen_errors = False

        # This script uses some heuristical tweaks to modify the documentation
        # of some analyzer options. As this code is fragile, we record the use
        # of these tweaks and report them if they become obsolete:
        self.unused_tweaks = [
            "escape star",
            "escape underline",
            "accepted values",
            "example file content",
        ]

    def record_use_of_tweak(self, tweak_name):
        try:
            self.unused_tweaks.remove(tweak_name)
        except ValueError:
            pass

    def replace_as_tweak(self, string, pattern, repl, tweak_name):
        res = string.replace(pattern, repl)
        if res != string:
            self.record_use_of_tweak(tweak_name)
        return res

    def report_error(self, msg):
        print("Error:", msg, file=sys.stderr)
        self.seen_errors = True

    def report_unexpected_char(self, s, pos):
        lines = (s[:pos] + "X").split("\n")
        lineno, col = (len(lines), len(lines[-1]))
        self.report_error(
            "unexpected character %r in AnalyzerOptions.def at line %d column %d"
            % (s[pos], lineno, col),
        )

    def report_unused_tweaks(self):
        if not self.unused_tweaks:
            return
        _is = " is" if len(self.unused_tweaks) == 1 else "s are"
        names = ", ".join(self.unused_tweaks)
        self.report_error(f"textual tweak{_is} unused in script: {names}")


err_handler = ErrorHandler()


def tokenize(s):
    result = []
    pos = 0
    while pos < len(s):
        for regex, kind in TOKENS:
            if m := regex.match(s, pos):
                if kind is not None:
                    result.append(Token(kind, m.group(0)))
                pos = m.end()
                break
        else:
            err_handler.report_unexpected_char(s, pos)
            pos += 1
    return result


def join_strings(tokens):
    result = []
    for tok in tokens:
        if tok.kind == TT.string and result and result[-1].kind == TT.string:
            # If this token is a string, and the previous non-ignored token is
            # also a string, then merge them into a single token. We need to
            # discard the closing " of the previous string and the opening " of
            # this string.
            prev = result.pop()
            result.append(Token(TT.string, prev.code[:-1] + tok.code[1:]))
        else:
            result.append(tok)
    return result


MacroCall = namedtuple("MacroCall", "name args")


class State(Enum):
    "States of the state machine used for parsing the macro calls."
    init = auto()
    after_ident = auto()
    before_arg = auto()
    after_arg = auto()


def get_calls(tokens, macro_names):
    state = State.init
    result = []
    current = None
    for tok in tokens:
        if state == State.init and tok.kind == TT.ident and tok.code in macro_names:
            current = MacroCall(tok.code, [])
            state = State.after_ident
        elif state == State.after_ident and tok == Token(TT.punct, "("):
            state = State.before_arg
        elif state == State.before_arg:
            if current is not None:
                current.args.append(tok)
                state = State.after_arg
        elif state == State.after_arg and tok.kind == TT.punct:
            if tok.code == ")":
                result.append(current)
                current = None
                state = State.init
            elif tok.code == ",":
                state = State.before_arg
        else:
            current = None
            state = State.init
    return result


# The information will be extracted from calls to these two macros:
# #define ANALYZER_OPTION(TYPE, NAME, CMDFLAG, DESC, DEFAULT_VAL)
# #define ANALYZER_OPTION_DEPENDS_ON_USER_MODE(TYPE, NAME, CMDFLAG, DESC,
#                                              SHALLOW_VAL, DEEP_VAL)

MACRO_NAMES_PARAMCOUNTS = {
    "ANALYZER_OPTION": 5,
    "ANALYZER_OPTION_DEPENDS_ON_USER_MODE": 6,
}


def string_value(tok):
    if tok.kind != TT.string:
        raise ValueError(f"expected a string token, got {tok.kind.name}")
    text = tok.code[1:-1]  # Remove quotes
    text = re.sub(r"\\(.)", r"\1", text)  # Resolve backslash escapes
    return text


def cmdflag_to_rst_title(cmdflag_tok):
    text = string_value(cmdflag_tok)
    underline = "-" * len(text)
    ref = f".. _analyzer-option-{text}:"

    return f"{ref}\n\n{text}\n{underline}\n\n"


def desc_to_rst_paragraphs(tok):
    desc = string_value(tok)

    # Escape some characters that have special meaning in RST:
    desc = err_handler.replace_as_tweak(desc, "*", r"\*", "escape star")
    desc = err_handler.replace_as_tweak(desc, "_", r"\_", "escape underline")

    # Many descriptions end with "Value: <list of accepted values>", which is
    # OK for a terse command line printout, but should be prettified for web
    # documentation.
    # Moreover, the option ctu-invocation-list shows some example file content
    # which is formatted as a preformatted block.
    paragraphs = [desc]
    extra = ""
    if m := re.search(r"(^|\s)Value:", desc):
        err_handler.record_use_of_tweak("accepted values")
        paragraphs = [desc[: m.start()], "Accepted values:" + desc[m.end() :]]
    elif m := re.search(r"\s*Example file.content:", desc):
        err_handler.record_use_of_tweak("example file content")
        paragraphs = [desc[: m.start()]]
        extra = "Example file content::\n\n  " + desc[m.end() :] + "\n\n"

    wrapped = [textwrap.fill(p, width=80) for p in paragraphs if p.strip()]

    return "\n\n".join(wrapped + [""]) + extra


def default_to_rst(tok):
    if tok.kind == TT.string:
        if tok.code == '""':
            return "(empty string)"
        return tok.code
    if tok.kind == TT.ident:
        return tok.code
    if tok.kind == TT.number:
        return tok.code.replace("'", "")
    raise ValueError(f"unexpected token as default value: {tok.kind.name}")


def defaults_to_rst_paragraph(defaults):
    strs = [default_to_rst(d) for d in defaults]

    if len(strs) == 1:
        return f"Default value: {strs[0]}\n\n"
    if len(strs) == 2:
        return (
            f"Default value: {strs[0]} (in shallow mode) / {strs[1]} (in deep mode)\n\n"
        )
    raise ValueError("unexpected count of default values: %d" % len(defaults))


def macro_call_to_rst_paragraphs(macro_call):
    try:
        arg_count = len(macro_call.args)
        param_count = MACRO_NAMES_PARAMCOUNTS[macro_call.name]
        if arg_count != param_count:
            raise ValueError(
                f"expected {param_count} arguments for {macro_call.name}, found {arg_count}"
            )

        _, _, cmdflag, desc, *defaults = macro_call.args

        return (
            cmdflag_to_rst_title(cmdflag)
            + desc_to_rst_paragraphs(desc)
            + defaults_to_rst_paragraph(defaults)
        )
    except ValueError as ve:
        err_handler.report_error(ve.args[0])
        return ""


def get_option_list(input_file):
    with open(input_file, encoding="utf-8") as f:
        contents = f.read()
    tokens = join_strings(tokenize(contents))
    macro_calls = get_calls(tokens, MACRO_NAMES_PARAMCOUNTS)

    result = ""
    for mc in macro_calls:
        result += macro_call_to_rst_paragraphs(mc)
    return result


p = argparse.ArgumentParser()
p.add_argument("--options-def", help="path to AnalyzerOptions.def")
p.add_argument("--template", help="template file")
p.add_argument("--out", help="output file")
opts = p.parse_args()

with open(opts.template, encoding="utf-8") as f:
    doc_template = f.read()

PLACEHOLDER = ".. OPTIONS_LIST_PLACEHOLDER\n"

rst_output = doc_template.replace(PLACEHOLDER, get_option_list(opts.options_def))

err_handler.report_unused_tweaks()

with open(opts.out, "w", newline="", encoding="utf-8") as f:
    f.write(rst_output)

if err_handler.seen_errors:
    sys.exit(1)
