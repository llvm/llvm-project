// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix
// DEFINE: %{codegen} = %clang -c %s -o %t.o -mllvm -debug-only=codegenaction 2>&1

// The codegen action should not clear the AST before running codegen.
// Codegen should otherwise clear the AST if ran without extraction.

// RUN: rm -rf %t.o %t.json
// RUN: %{codegen} | grep "Clearing AST"

// RUN: rm -rf %t.o %t.json
// RUN: %{codegen} \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-tu-summary-file=%t.json \
// RUN: | not grep "Clearing AST"

void empty() {}
