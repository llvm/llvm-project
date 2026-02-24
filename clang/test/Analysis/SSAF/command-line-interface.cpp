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

// NOT-MATCHING-THE-PATTERN: fatal error: failed to parse the value of '--ssaf-tu-summary-file=foobar' the value must follow the '<path>.<format>' pattern
// UNKNOWN-FILE-FORMAT:      fatal error: unknown output summary file format 'unknownfmt' specified by '--ssaf-tu-summary-file={{.+}}.ssaf.unknownfmt'
// NO-EXTRACTORS-ENABLED:    fatal error: must enable some summary extractors using the '--ssaf-extract-summaries=' option
// NO-EXTRACTOR-WITH-NAME:   fatal error: no summary extractor was registered with name: extractor1
// NO-EXTRACTORS-WITH-NAME:  fatal error: no summary extractors were registered with name: extractor1, extractor2
