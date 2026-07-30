// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix

// RUN: %clang     --help 2>&1 | %{filecheck}=HELP
// RUN: %clang_cc1 --help 2>&1 | %{filecheck}=HELP

// HELP:       --ssaf-compilation-unit-id=<id>
// HELP-NEXT:    Stable identifier used as the CompilationUnit namespace name of every produced SSAF TU summary. Required when '--ssaf-tu-summary-file=' is set.
// HELP-NEXT:  --ssaf-extract-summaries=<summary-names>
// HELP-NEXT:    Comma-separated list of summary names to extract
// HELP-NEXT:  --ssaf-global-scope-analysis-result=<path>.<format>
// HELP-NEXT:    Path to the WPASuite file containing the whole-program analysis result consumed by the source transformation. The extension selects which file format to use.
// HELP-NEXT:  --ssaf-include-local-entities
// HELP-NEXT:    Include block-scope (function-local) declarations in extracted SSAF summaries. By default they are omitted.
// HELP-NEXT:  --ssaf-list-extractors  Display the list of available SSAF summary extractors
// HELP-NEXT:  --ssaf-list-formats     Display the list of available SSAF serialization formats
// HELP-NEXT:  --ssaf-no-extract-from-system-headers
// HELP-NEXT:    Skip declarations in system headers during SSAF summary extraction
// HELP-NEXT:  --ssaf-source-transformation=<name>
// HELP-NEXT:    Name of the SSAF source transformation to run. Exactly one transformation per invocation.
// HELP-NEXT:  --ssaf-src-edit-file=<path>
// HELP-NEXT:    Output file for the source edits produced by the source transformation. The output is a YAML document compatible with 'clang-apply-replacements'.
// HELP-NEXT:  --ssaf-transformation-report-file=<path>
// HELP-NEXT:    Output file for the transformation report produced by the source transformation. The output is a SARIF JSON document.
// HELP-NEXT:  --ssaf-tu-summary-file=<path>.<format>
// HELP-NEXT:    The output file for the extracted summaries. The extension selects which file format to use.

// FIXME: --ssaf-list-{extractors,formats} only work with the `clang` driver.
// RUN: %clang --ssaf-list-extractors 2>&1 | %{filecheck}=LIST-EXTRACTORS
// LIST-EXTRACTORS: OVERVIEW: Available SSAF summary extractors:

// RUN: %clang --ssaf-list-formats 2>&1 | %{filecheck}=LIST-FORMATS
// LIST-FORMATS: OVERVIEW: Available SSAF serialization formats:
// LIST-FORMATS:   json - JSON serialization format

// RUN: %clang --ssaf-list-extractors --ssaf-list-formats 2>&1 | %{filecheck}=LIST-EXTRACTORS,LIST-FORMATS
