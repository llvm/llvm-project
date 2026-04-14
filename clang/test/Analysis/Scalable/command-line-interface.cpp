// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix

// The flags should behave the same way on the clang driver and also on CC1.

// RUN: not %clang     -fsyntax-only %s --ssaf-tu-summary-file=foobar             2>&1 | %{filecheck}=NOT-MATCHING-THE-PATTERN
// RUN: not %clang_cc1 -fsyntax-only %s --ssaf-tu-summary-file=foobar             2>&1 | %{filecheck}=NOT-MATCHING-THE-PATTERN
// RUN: not %clang     -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.unknownfmt 2>&1 | %{filecheck}=UNKNOWN-FILE-FORMAT
// RUN: not %clang_cc1 -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.unknownfmt 2>&1 | %{filecheck}=UNKNOWN-FILE-FORMAT
// RUN: not %clang     -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json       2>&1 | %{filecheck}=NO-EXTRACTORS-ENABLED
// RUN: not %clang_cc1 -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json       2>&1 | %{filecheck}=NO-EXTRACTORS-ENABLED
// RUN: not %clang     -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json --ssaf-extract-summaries=extractor1            2>&1 | %{filecheck}=NO-EXTRACTOR-WITH-NAME
// RUN: not %clang_cc1 -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json --ssaf-extract-summaries=extractor1            2>&1 | %{filecheck}=NO-EXTRACTOR-WITH-NAME
// RUN: not %clang     -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json --ssaf-extract-summaries=extractor1,extractor2 2>&1 | %{filecheck}=NO-EXTRACTORS-WITH-NAME
// RUN: not %clang_cc1 -fsyntax-only %s --ssaf-tu-summary-file=%t.ssaf.json --ssaf-extract-summaries=extractor1,extractor2 2>&1 | %{filecheck}=NO-EXTRACTORS-WITH-NAME

void empty() {}

// NOT-MATCHING-THE-PATTERN: error: failed to parse the value of '--ssaf-tu-summary-file=foobar' the value must follow the '<path>.<format>' pattern [-Wscalable-static-analysis-framework]
// UNKNOWN-FILE-FORMAT:      error: unknown output summary file format 'unknownfmt' specified by '--ssaf-tu-summary-file={{.+}}.ssaf.unknownfmt' [-Wscalable-static-analysis-framework]
// NO-EXTRACTORS-ENABLED:    error: must enable some summary extractors using the '--ssaf-extract-summaries=' option [-Wscalable-static-analysis-framework]
// NO-EXTRACTOR-WITH-NAME:   error: no summary extractor was registered with name: extractor1 [-Wscalable-static-analysis-framework]
// NO-EXTRACTORS-WITH-NAME:  error: no summary extractors were registered with name: extractor1, extractor2 [-Wscalable-static-analysis-framework]
