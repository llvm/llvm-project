// RUN: rm -rf %t
// RUN: mkdir -p %t

// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-compilation-unit-id=test-cu \
// RUN:   --ssaf-tu-summary-file=%t/cg.default.json
// RUN: %clang_cc1 -fsyntax-only %s \
// RUN:   --ssaf-extract-summaries=CallGraph \
// RUN:   --ssaf-include-local-entities \
// RUN:   --ssaf-compilation-unit-id=test-cu \
// RUN:   --ssaf-tu-summary-file=%t/cg.with_locals.json

// FIXME: The next line should assert 1 count because the lambda call operator
//        should be included only if requested.
// RUN: cat %t/cg.default.json     | grep '"entity_id":' | count 2
// RUN: cat %t/cg.with_locals.json | grep '"entity_id":' | count 2

void caller() {
  auto local = []{};
  local();
}
