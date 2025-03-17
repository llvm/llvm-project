// DEFINE: %{custom-call-yaml} = custom-call: 'm callExpr().bind(\"Custom message\")'
//
// DEFINE: %{custom-let-call-yaml} = custom-let-call: \"         \
// DEFINE:     let expr varDecl(                                 \
// DEFINE:       hasType(asString(\\\"long long\\\")),           \
// DEFINE:       hasTypeLoc(typeLoc().bind(\\\"Let message\\\")) \
// DEFINE:     ) \n                                              \
// DEFINE:     match expr\"
//
// DEFINE: %{full-config} = "{ClangQueryChecks: {%{custom-call-yaml},%{custom-let-call-yaml}}}"

//Check single match expression
// RUN: clang-tidy %s -checks='-*, custom-*'                  \
// RUN:   -config="{ClangQueryChecks: {%{custom-call-yaml}}}" \
// RUN:   -- | FileCheck %s -check-prefix=CHECK-CUSTOM-CALL

void a() {
}

// CHECK-CUSTOM-CALL: warning: Custom message [custom-call]
// CHECK-CUSTOM-CALL-NEXT: a();{{$}}
void b() {
    a();
}

//Check let with match expression
// RUN: clang-tidy %s -checks='-*, custom-*'                      \
// RUN:   -config="{ClangQueryChecks: {%{custom-let-call-yaml}}}" \
// RUN:   -- | FileCheck %s -check-prefix=CHECK-CUSTOM-LET
void c() {
    // CHECK-CUSTOM-LET: warning: Let message [custom-let-call]
    // CHECK-CUSTOM-LET-NEXT: long long test_long_long = 0;{{$}}
    long long test_long_long_nolint = 0; //NOLINT(custom-let-call)
    long long test_long_long = 0;
}

//Check multiple checks in one config
// RUN: clang-tidy %s -checks='-*, custom-*' \
// RUN:   -config=%{full-config}             \
// RUN:   -- | FileCheck %s -check-prefixes=CHECK-CUSTOM-CALL,CHECK-CUSTOM-LET

//Check multiple checks in one config but only one enabled
// RUN: clang-tidy %s -checks='-*, custom-call' \
// RUN:   -config=%{full-config}                \
// RUN:   -- | FileCheck %s -check-prefixes=CHECK-CUSTOM-CALL --implicit-check-not warning:

//Check config dump
// RUN: clang-tidy -dump-config -checks='-*, custom-*' \
// RUN:   -config=%{full-config}                       \
// RUN:   -- | FileCheck %s -check-prefix=CHECK-CONFIG
// CHECK-CONFIG: ClangQueryChecks:
// CHECK-CONFIG-DAG: custom-let-call:
// CHECK-CONFIG-DAG: custom-call:  |{{$[[:space:]]}} m callExpr().bind("Custom message")
