#!/usr/bin/env python3
#
# ===- add_new_check.py - clang-tidy check generator ---------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===-----------------------------------------------------------------------===#

import argparse
import io
import itertools
import os
import re
import sys
import textwrap

# FIXME Python 3.9: Replace typing.Tuple with builtins.tuple.
from typing import Optional, Tuple, Match


# Adapts the module's CMakelist file. Returns 'True' if it could add a new
# entry and 'False' if the entry already existed.
def adapt_cmake(module_path: str, check_name_camel: str) -> bool:
    filename = os.path.join(module_path, "CMakeLists.txt")

    # The documentation files are encoded using UTF-8, however on Windows the
    # default encoding might be different (e.g. CP-1252). To make sure UTF-8 is
    # always used, use `io.open(filename, mode, encoding='utf8')` for reading and
    # writing files here and elsewhere.
    with io.open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()

    cpp_file = check_name_camel + ".cpp"

    # Figure out whether this check already exists.
    for line in lines:
        if line.strip() == cpp_file:
            return False

    print("Updating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        cpp_found = False
        file_added = False
        for line in lines:
            cpp_line = line.strip().endswith(".cpp")
            if (not file_added) and (cpp_line or cpp_found):
                cpp_found = True
                if (line.strip() > cpp_file) or (not cpp_line):
                    f.write("  " + cpp_file + "\n")
                    file_added = True
            f.write(line)

    return True


# Adds a header for the new check.
def write_header(
    module_path: str,
    module: str,
    namespace: str,
    check_name: str,
    check_name_camel: str,
    description: str,
    lang_restrict: str,
) -> None:
    wrapped_desc = "\n".join(
        textwrap.wrap(
            description, width=80, initial_indent="/// ", subsequent_indent="/// "
        )
    )
    if lang_restrict:
        override_supported = """
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return %s;
  }""" % (
            lang_restrict % {"lang": "LangOpts"}
        )
    else:
        override_supported = ""
    filename = os.path.join(module_path, check_name_camel) + ".h"
    print("Creating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        header_guard = (
            "LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_"
            + module.upper()
            + "_"
            + check_name_camel.upper()
            + "_H"
        )
        f.write("//===--- ")
        f.write(os.path.basename(filename))
        f.write(" - clang-tidy ")
        f.write("-" * max(0, 42 - len(os.path.basename(filename))))
        f.write("*- C++ -*-===//")
        f.write(
            """
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef %(header_guard)s
#define %(header_guard)s

#include "../ClangTidyCheck.h"

namespace clang::tidy::%(namespace)s {

%(description)s
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/%(module)s/%(check_name)s.html
class %(check_name_camel)s : public ClangTidyCheck {
public:
  %(check_name_camel)s(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;%(override_supported)s
};

} // namespace clang::tidy::%(namespace)s

#endif // %(header_guard)s
"""
            % {
                "header_guard": header_guard,
                "check_name_camel": check_name_camel,
                "check_name": check_name,
                "module": module,
                "namespace": namespace,
                "description": wrapped_desc,
                "override_supported": override_supported,
            }
        )


# Adds the implementation of the new check.
def write_implementation(
    module_path: str, module: str, namespace: str, check_name_camel: str
) -> None:
    filename = os.path.join(module_path, check_name_camel) + ".cpp"
    print("Creating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        f.write("//===--- ")
        f.write(os.path.basename(filename))
        f.write(" - clang-tidy ")
        f.write("-" * max(0, 51 - len(os.path.basename(filename))))
        f.write("-===//")
        f.write(
            """
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "%(check_name)s.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::%(namespace)s {

void %(check_name)s::registerMatchers(MatchFinder *Finder) {
  // FIXME: Add matchers.
  Finder->addMatcher(functionDecl().bind("x"), this);
}

void %(check_name)s::check(const MatchFinder::MatchResult &Result) {
  // FIXME: Add callback implementation.
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("x");
  if (!MatchedDecl->getIdentifier() || MatchedDecl->getName().starts_with("awesome_"))
    return;
  diag(MatchedDecl->getLocation(), "function %%0 is insufficiently awesome")
      << MatchedDecl
      << FixItHint::CreateInsertion(MatchedDecl->getLocation(), "awesome_");
  diag(MatchedDecl->getLocation(), "insert 'awesome'", DiagnosticIDs::Note);
}

} // namespace clang::tidy::%(namespace)s
"""
            % {"check_name": check_name_camel, "module": module, "namespace": namespace}
        )


# Returns the source filename that implements the module.
def get_module_filename(module_path: str, module: str) -> str:
    modulecpp = list(
        filter(
            lambda p: p.lower() == module.lower() + "tidymodule.cpp",
            os.listdir(module_path),
        )
    )[0]
    return os.path.join(module_path, modulecpp)


# Modifies the module to include the new check.
def adapt_module(
    module_path: str, module: str, check_name: str, check_name_camel: str
) -> None:
    filename = get_module_filename(module_path, module)
    with io.open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()

    print("Updating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        header_added = False
        header_found = False
        check_added = False
        check_fq_name = module + "-" + check_name
        check_decl = (
            "    CheckFactories.registerCheck<"
            + check_name_camel
            + '>(\n        "'
            + check_fq_name
            + '");\n'
        )

        lines_iter = iter(lines)
        try:
            while True:
                line = next(lines_iter)
                if not header_added:
                    match = re.search('#include "(.*)"', line)
                    if match:
                        header_found = True
                        if match.group(1) > check_name_camel:
                            header_added = True
                            f.write('#include "' + check_name_camel + '.h"\n')
                    elif header_found:
                        header_added = True
                        f.write('#include "' + check_name_camel + '.h"\n')

                if not check_added:
                    if line.strip() == "}":
                        check_added = True
                        f.write(check_decl)
                    else:
                        match = re.search(
                            r'registerCheck<(.*)> *\( *(?:"([^"]*)")?', line
                        )
                        prev_line = None
                        if match:
                            current_check_name = match.group(2)
                            if current_check_name is None:
                                # If we didn't find the check name on this line, look on the
                                # next one.
                                prev_line = line
                                line = next(lines_iter)
                                match = re.search(' *"([^"]*)"', line)
                                if match:
                                    current_check_name = match.group(1)
                            assert current_check_name
                            if current_check_name > check_fq_name:
                                check_added = True
                                f.write(check_decl)
                            if prev_line:
                                f.write(prev_line)
                f.write(line)
        except StopIteration:
            pass


# Adds a release notes entry.
def add_release_notes(
    module_path: str, module: str, check_name: str, description: str
) -> None:
    wrapped_desc = "\n".join(
        textwrap.wrap(
            description, width=80, initial_indent="  ", subsequent_indent="  "
        )
    )
    check_name_dashes = module + "-" + check_name
    filename = os.path.normpath(
        os.path.join(module_path, "../../docs/ReleaseNotes.rst")
    )
    with io.open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()

    lineMatcher = re.compile("New checks")
    nextSectionMatcher = re.compile("New check aliases")
    checkMatcher = re.compile("- New :doc:`(.*)")

    print("Updating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        note_added = False
        header_found = False
        add_note_here = False

        for line in lines:
            if not note_added:
                match = lineMatcher.match(line)
                match_next = nextSectionMatcher.match(line)
                match_check = checkMatcher.match(line)
                if match_check:
                    last_check = match_check.group(1)
                    if last_check > check_name_dashes:
                        add_note_here = True

                if match_next:
                    add_note_here = True

                if match:
                    header_found = True
                    f.write(line)
                    continue

                if line.startswith("^^^^"):
                    f.write(line)
                    continue

                if header_found and add_note_here:
                    if not line.startswith("^^^^"):
                        f.write(
                            """- New :doc:`%s
  <clang-tidy/checks/%s/%s>` check.

%s

"""
                            % (check_name_dashes, module, check_name, wrapped_desc)
                        )
                        note_added = True

            f.write(line)


# Adds a test for the check.
def write_test(
    module_path: str,
    module: str,
    check_name: str,
    test_extension: str,
    test_standard: Optional[str],
) -> None:
    test_standard = f"-std={test_standard}-or-later " if test_standard else ""
    check_name_dashes = module + "-" + check_name
    filename = os.path.normpath(
        os.path.join(
            module_path,
            "..",
            "..",
            "test",
            "clang-tidy",
            "checkers",
            module,
            check_name + "." + test_extension,
        )
    )
    print("Creating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        f.write(
            """// RUN: %%check_clang_tidy %(standard)s%%s %(check_name_dashes)s %%t

// FIXME: Add something that triggers the check here.
void f();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'f' is insufficiently awesome [%(check_name_dashes)s]

// FIXME: Verify the applied fix.
//   * Make the CHECK patterns specific enough and try to make verified lines
//     unique to avoid incorrect matches.
//   * Use {{}} for regular expressions.
// CHECK-FIXES: {{^}}void awesome_f();{{$}}

// FIXME: Add something that doesn't trigger the check here.
void awesome_f2();
"""
            % {"check_name_dashes": check_name_dashes, "standard": test_standard}
        )


def get_actual_filename(dirname: str, filename: str) -> str:
    if not os.path.isdir(dirname):
        return ""
    name = os.path.join(dirname, filename)
    if os.path.isfile(name):
        return name
    caselessname = filename.lower()
    for file in os.listdir(dirname):
        if file.lower() == caselessname:
            return os.path.join(dirname, file)
    return ""


# Recreates the list of checks in the docs/clang-tidy/checks directory.
def update_checks_list(clang_tidy_path: str) -> None:
    docs_dir = os.path.join(clang_tidy_path, "../docs/clang-tidy/checks")
    filename = os.path.normpath(os.path.join(docs_dir, "list.rst"))
    # Read the content of the current list.rst file
    with io.open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()
    # Get all existing docs
    doc_files = []
    for subdir in filter(
        lambda s: os.path.isdir(os.path.join(docs_dir, s)), os.listdir(docs_dir)
    ):
        for file in filter(
            lambda s: s.endswith(".rst"), os.listdir(os.path.join(docs_dir, subdir))
        ):
            doc_files.append((subdir, file))
    doc_files.sort()

    # We couldn't find the source file from the check name, so try to find the
    # class name that corresponds to the check in the module file.
    def filename_from_module(module_name: str, check_name: str) -> str:
        module_path = os.path.join(clang_tidy_path, module_name)
        if not os.path.isdir(module_path):
            return ""
        module_file = get_module_filename(module_path, module_name)
        if not os.path.isfile(module_file):
            return ""
        with io.open(module_file, "r") as f:
            code = f.read()
            full_check_name = module_name + "-" + check_name
            name_pos = code.find('"' + full_check_name + '"')
            if name_pos == -1:
                return ""
            stmt_end_pos = code.find(";", name_pos)
            if stmt_end_pos == -1:
                return ""
            stmt_start_pos = code.rfind(";", 0, name_pos)
            if stmt_start_pos == -1:
                stmt_start_pos = code.rfind("{", 0, name_pos)
            if stmt_start_pos == -1:
                return ""
            stmt = code[stmt_start_pos + 1 : stmt_end_pos]
            matches = re.search(r'registerCheck<([^>:]*)>\(\s*"([^"]*)"\s*\)', stmt)
            if matches and matches[2] == full_check_name:
                class_name = matches[1]
                if "::" in class_name:
                    parts = class_name.split("::")
                    class_name = parts[-1]
                    class_path = os.path.join(
                        clang_tidy_path, module_name, "..", *parts[0:-1]
                    )
                else:
                    class_path = os.path.join(clang_tidy_path, module_name)
                return get_actual_filename(class_path, class_name + ".cpp")

        return ""

    # Examine code looking for a c'tor definition to get the base class name.
    def get_base_class(code: str, check_file: str) -> str:
        check_class_name = os.path.splitext(os.path.basename(check_file))[0]
        ctor_pattern = check_class_name + r"\([^:]*\)\s*:\s*([A-Z][A-Za-z0-9]*Check)\("
        matches = re.search(r"\s+" + check_class_name + "::" + ctor_pattern, code)

        # The constructor might be inline in the header.
        if not matches:
            header_file = os.path.splitext(check_file)[0] + ".h"
            if not os.path.isfile(header_file):
                return ""
            with io.open(header_file, encoding="utf8") as f:
                code = f.read()
            matches = re.search(" " + ctor_pattern, code)

        if matches and matches[1] != "ClangTidyCheck":
            return matches[1]
        return ""

    # Some simple heuristics to figure out if a check has an autofix or not.
    def has_fixits(code: str) -> bool:
        for needle in [
            "FixItHint",
            "ReplacementText",
            "fixit",
            "TransformerClangTidyCheck",
        ]:
            if needle in code:
                return True
        return False

    # Try to figure out of the check supports fixits.
    def has_auto_fix(check_name: str) -> str:
        dirname, _, check_name = check_name.partition("-")

        check_file = get_actual_filename(
            os.path.join(clang_tidy_path, dirname),
            get_camel_check_name(check_name) + ".cpp",
        )
        if not os.path.isfile(check_file):
            # Some older checks don't end with 'Check.cpp'
            check_file = get_actual_filename(
                os.path.join(clang_tidy_path, dirname),
                get_camel_name(check_name) + ".cpp",
            )
            if not os.path.isfile(check_file):
                # Some checks aren't in a file based on the check name.
                check_file = filename_from_module(dirname, check_name)
                if not check_file or not os.path.isfile(check_file):
                    return ""

        with io.open(check_file, encoding="utf8") as f:
            code = f.read()
            if has_fixits(code):
                return ' "Yes"'

        base_class = get_base_class(code, check_file)
        if base_class:
            base_file = os.path.join(clang_tidy_path, dirname, base_class + ".cpp")
            if os.path.isfile(base_file):
                with io.open(base_file, encoding="utf8") as f:
                    code = f.read()
                    if has_fixits(code):
                        return ' "Yes"'

        return ""

    def process_doc(doc_file: Tuple[str, str]) -> Tuple[str, Optional[Match[str]]]:
        check_name = doc_file[0] + "-" + doc_file[1].replace(".rst", "")

        with io.open(os.path.join(docs_dir, *doc_file), "r", encoding="utf8") as doc:
            content = doc.read()
            match = re.search(".*:orphan:.*", content)

            if match:
                # Orphan page, don't list it.
                return "", None

            match = re.search(r".*:http-equiv=refresh: \d+;URL=(.*).html(.*)", content)
            # Is it a redirect?
            return check_name, match

    def format_link(doc_file: Tuple[str, str]) -> str:
        check_name, match = process_doc(doc_file)
        if not match and check_name and not check_name.startswith("clang-analyzer-"):
            return "   :doc:`%(check_name)s <%(module)s/%(check)s>`,%(autofix)s\n" % {
                "check_name": check_name,
                "module": doc_file[0],
                "check": doc_file[1].replace(".rst", ""),
                "autofix": has_auto_fix(check_name),
            }
        else:
            return ""

    def format_link_alias(doc_file: Tuple[str, str]) -> str:
        check_name, match = process_doc(doc_file)
        if (match or (check_name.startswith("clang-analyzer-"))) and check_name:
            module = doc_file[0]
            check_file = doc_file[1].replace(".rst", "")
            if (
                not match
                or match.group(1) == "https://clang.llvm.org/docs/analyzer/checkers"
            ):
                title = "Clang Static Analyzer " + check_file
                # Preserve the anchor in checkers.html from group 2.
                target = "" if not match else match.group(1) + ".html" + match.group(2)
                autofix = ""
                ref_begin = ""
                ref_end = "_"
            else:
                redirect_parts = re.search(r"^\.\./([^/]*)/([^/]*)$", match.group(1))
                assert redirect_parts
                title = redirect_parts[1] + "-" + redirect_parts[2]
                target = redirect_parts[1] + "/" + redirect_parts[2]
                autofix = has_auto_fix(title)
                ref_begin = ":doc:"
                ref_end = ""

            if target:
                # The checker is just a redirect.
                return (
                    "   :doc:`%(check_name)s <%(module)s/%(check_file)s>`, %(ref_begin)s`%(title)s <%(target)s>`%(ref_end)s,%(autofix)s\n"
                    % {
                        "check_name": check_name,
                        "module": module,
                        "check_file": check_file,
                        "target": target,
                        "title": title,
                        "autofix": autofix,
                        "ref_begin": ref_begin,
                        "ref_end": ref_end,
                    }
                )
            else:
                # The checker is just a alias without redirect.
                return (
                    "   :doc:`%(check_name)s <%(module)s/%(check_file)s>`, %(title)s,%(autofix)s\n"
                    % {
                        "check_name": check_name,
                        "module": module,
                        "check_file": check_file,
                        "target": target,
                        "title": title,
                        "autofix": autofix,
                    }
                )
        return ""

    checks = map(format_link, doc_files)
    checks_alias = map(format_link_alias, doc_files)

    print("Updating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        for line in lines:
            f.write(line)
            if line.strip() == ".. csv-table::":
                # We dump the checkers
                f.write('   :header: "Name", "Offers fixes"\n\n')
                f.writelines(checks)
                # and the aliases
                f.write("\nCheck aliases\n-------------\n\n")
                f.write(".. csv-table::\n")
                f.write('   :header: "Name", "Redirect", "Offers fixes"\n\n')
                f.writelines(checks_alias)
                break


# Adds a documentation for the check.
def write_docs(module_path: str, module: str, check_name: str) -> None:
    check_name_dashes = module + "-" + check_name
    filename = os.path.normpath(
        os.path.join(
            module_path, "../../docs/clang-tidy/checks/", module, check_name + ".rst"
        )
    )
    print("Creating %s..." % filename)
    with io.open(filename, "w", encoding="utf8", newline="\n") as f:
        f.write(
            """.. title:: clang-tidy - %(check_name_dashes)s

%(check_name_dashes)s
%(underline)s

FIXME: Describe what patterns does the check detect and why. Give examples.
"""
            % {
                "check_name_dashes": check_name_dashes,
                "underline": "=" * len(check_name_dashes),
            }
        )


def get_camel_name(check_name: str) -> str:
    return "".join(map(lambda elem: elem.capitalize(), check_name.split("-")))


def get_camel_check_name(check_name: str) -> str:
    return get_camel_name(check_name) + "Check"


def main() -> None:
    language_to_extension = {
        "c": "c",
        "c++": "cpp",
        "objc": "m",
        "objc++": "mm",
    }
    cpp_language_to_requirements = {
        "c++98": "CPlusPlus",
        "c++11": "CPlusPlus11",
        "c++14": "CPlusPlus14",
        "c++17": "CPlusPlus17",
        "c++20": "CPlusPlus20",
        "c++23": "CPlusPlus23",
        "c++26": "CPlusPlus26",
    }
    c_language_to_requirements = {
        "c99": None,
        "c11": "C11",
        "c17": "C17",
        "c23": "C23",
        "c27": "C2Y",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-docs",
        action="store_true",
        help="just update the list of documentation files, then exit",
    )
    parser.add_argument(
        "--language",
        help="language to use for new check (defaults to c++)",
        choices=language_to_extension.keys(),
        default=None,
        metavar="LANG",
    )
    parser.add_argument(
        "--description",
        "-d",
        help="short description of what the check does",
        default="FIXME: Write a short description",
        type=str,
    )
    parser.add_argument(
        "--standard",
        help="Specify a specific version of the language",
        choices=list(
            itertools.chain(
                cpp_language_to_requirements.keys(), c_language_to_requirements.keys()
            )
        ),
        default=None,
    )
    parser.add_argument(
        "module",
        nargs="?",
        help="module directory under which to place the new tidy check (e.g., misc)",
    )
    parser.add_argument(
        "check", nargs="?", help="name of new tidy check to add (e.g. foo-do-the-stuff)"
    )
    args = parser.parse_args()

    if args.update_docs:
        update_checks_list(os.path.dirname(sys.argv[0]))
        return

    if not args.module or not args.check:
        print("Module and check must be specified.")
        parser.print_usage()
        return

    module = args.module
    check_name = args.check
    check_name_camel = get_camel_check_name(check_name)
    if check_name.startswith(module):
        print(
            'Check name "%s" must not start with the module "%s". Exiting.'
            % (check_name, module)
        )
        return
    clang_tidy_path = os.path.dirname(sys.argv[0])
    module_path = os.path.join(clang_tidy_path, module)

    if not adapt_cmake(module_path, check_name_camel):
        return

    # Map module names to namespace names that don't conflict with widely used top-level namespaces.
    if module == "llvm":
        namespace = module + "_check"
    else:
        namespace = module

    description = args.description
    if not description.endswith("."):
        description += "."

    language = args.language

    if args.standard:
        if args.standard in cpp_language_to_requirements:
            if language and language != "c++":
                raise ValueError("C++ standard chosen when language is not C++")
            language = "c++"
        elif args.standard in c_language_to_requirements:
            if language and language != "c":
                raise ValueError("C standard chosen when language is not C")
            language = "c"

    if not language:
        language = "c++"

    language_restrict = None

    if language == "c":
        language_restrict = "!%(lang)s.CPlusPlus"
        extra = c_language_to_requirements.get(args.standard, None)
        if extra:
            language_restrict += f" && %(lang)s.{extra}"
    elif language == "c++":
        language_restrict = (
            f"%(lang)s.{cpp_language_to_requirements.get(args.standard, 'CPlusPlus')}"
        )
    elif language in ["objc", "objc++"]:
        language_restrict = "%(lang)s.ObjC"
    else:
        raise ValueError(f"Unsupported language '{language}' was specified")

    write_header(
        module_path,
        module,
        namespace,
        check_name,
        check_name_camel,
        description,
        language_restrict,
    )
    write_implementation(module_path, module, namespace, check_name_camel)
    adapt_module(module_path, module, check_name, check_name_camel)
    add_release_notes(module_path, module, check_name, description)
    test_extension = language_to_extension[language]
    write_test(module_path, module, check_name, test_extension, args.standard)
    write_docs(module_path, module, check_name)
    update_checks_list(clang_tidy_path)
    print("Done. Now it's your turn!")


if __name__ == "__main__":
    main()
