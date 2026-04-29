// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix

// RUN: %clang     --help 2>&1 | %{filecheck}=HELP
// RUN: %clang_cc1 --help 2>&1 | %{filecheck}=HELP

// HELP:       --ssaf-apply-source-pass=<analysis-names>
// HELP-NEXT:    Comma-separated list of source-pass analysis names to apply
// HELP-NEXT:  --ssaf-extract-summaries=<summary-names>
// HELP-NEXT:    Comma-separated list of summary names to extract
// HELP-NEXT:  --ssaf-list-extractors  Display the list of available SSAF summary extractors
// HELP-NEXT:  --ssaf-list-formats     Display the list of available SSAF serialization formats
// HELP-NEXT:  --ssaf-load-wpa-result=<path>.<format>
// HELP-NEXT:    The WPA result file to load. The extension selects which file format to use.
// HELP-NEXT:  --ssaf-tu-summary-file=<path>.<format>
// HELP-NEXT:    The output file for the extracted summaries. The extension selects which file format to use.

// FIXME: --ssaf-list-{extractors,formats} only work with the `clang` driver.
// RUN: %clang --ssaf-list-extractors 2>&1 | %{filecheck}=LIST-EXTRACTORS
// LIST-EXTRACTORS: OVERVIEW: Available SSAF summary extractors:

// RUN: %clang --ssaf-list-formats 2>&1 | %{filecheck}=LIST-FORMATS
// LIST-FORMATS: OVERVIEW: Available SSAF serialization formats:
// LIST-FORMATS:   json - JSON serialization format

// RUN: %clang --ssaf-list-extractors --ssaf-list-formats 2>&1 | %{filecheck}=LIST-EXTRACTORS,LIST-FORMATS
