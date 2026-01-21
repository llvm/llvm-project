// RUN: %clang -fsyntax-only -Wall -Wextra -fdiagnostics-format=sarif %s > %t 2>&1 || true
// RUN: FileCheck -dump-input=always %s --input-file=%t

// FIXME: this test is incredibly fragile because the `main()` function
// must be on line 12 in order for the CHECK lines to get the correct line
// number values.
//
// So these comment lines are being used to ensure the code below happens
// to work properly for the test coverage, which as you can imagine, is not
// the best way to structure the test. We really need to introduce a better
// tool than FileCheck for diff'ing JSON output like SARIF.
void main() {
  int i = hello;

  float test = 1a.0;

  if (true)
    bool Yes = true;
    return;

  bool j = hi;
}
}

struct t1 { };
void f1(t1 x, t1 y) {
    x + y;
}

// CHECK: warning: diagnostic formatting in SARIF mode is currently unstable [-Wsarif-format-unstable]
// CHECK: {"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json","runs":[{"artifacts":[{"length":
// Omit exact length of this file
// CHECK: ,"location":{"index":0,"uri":"file://
// Omit filepath to llvm project directory
// CHECK: test/Frontend/sarif-diagnostics.cpp"},"mimeType":"text/plain","roles":["resultFile"]}],"columnKind":"unicodeCodePoints",
// CHECK: "invocations": 
// CHECK: [{"commandLine":"{{.*}}","executionSuccessful":false,"exitCode":1,"toolExecutionNotifications":[{"level":"note","message":{"text":"2 warnings and 6 errors generated.{{[^"]*}}"}}]}],
// CHECK: "results":
// CHECK: [{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":1,"startColumn":1,"startLine":12}}}],"message":{"text":"'main' must return 'int'"},"ruleId":"{{[0-9]+}}","ruleIndex":0},
// CHECK: {"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":16,"endLine":13,"startColumn":11,"startLine":13}}},{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":11,"startColumn":11,"startLine":13}}}],"message":{"text":"use of undeclared identifier 'hello'"},"ruleId":"{{[0-9]+}}","ruleIndex":1},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":
// CHECK: {"index":0,"uri":"file://{{.+}}"},"region":{"endColumn":17,"startColumn":17,"startLine":15}}}],"message":{"text":"invalid digit 'a' in decimal constant"},"ruleId":"{{[0-9]+}}","ruleIndex":2},{"level":"warning","locations":[{"physicalLocation":{"artifactLocation":
// CHECK: {"index":0,"uri":"file://{{.+}}"},"region":{"endColumn":5,"startColumn":5,"startLine":19}}}],"message":{"text":"misleading indentation; statement is not part of the previous 'if'"},"relatedLocations":[{"message":{"text":"previous statement is here"},"physicalLocation":{"artifactLocation":
// CHECK: {"index":0,"uri":"file://{{.+}}"},"region":{"endColumn":3,"startColumn":3,"startLine":17}},"properties":{"nestingLevel":1}}],"ruleId":"{{[0-9]+}}","ruleIndex":3},{"level":"warning","locations":[{"physicalLocation":{"artifactLocation":
// CHECK: {"index":0,"uri":"file://{{.+}}"},"region":{"endColumn":13,"endLine":18,"startColumn":10,"startLine":18}}},{"physicalLocation":{"artifactLocation":
// CHECK: {"index":0,"uri":"file://{{.+}}"},"region":{"endColumn":10,"startColumn":10,"startLine":18}}}],"message":{"text":"unused variable 'Yes'"},"ruleId":"{{[0-9]+}}","ruleIndex":4},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":14,"endLine":21,"startColumn":12,"startLine":21}}},{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":12,"startColumn":12,"startLine":21}}}],"message":{"text":"use of undeclared identifier 'hi'"},"ruleId":"{{[0-9]+}}","ruleIndex":5},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":1,"startColumn":1,"startLine":23}}}],"message":{"text":"extraneous closing brace ('}')"},"ruleId":"{{[0-9]+}}","ruleIndex":6},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":6,"endLine":27,"startColumn":5,"startLine":27}}},{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":10,"endLine":27,"startColumn":9,"startLine":27}}},{"physicalLocation":{"artifactLocation":{"index":0,"uri":"file://
// CHECK: {"endColumn":7,"startColumn":7,"startLine":27}}}],"message":{"text":"invalid operands to binary expression ('t1' and 't1')"},"ruleId":"{{[0-9]+}}","ruleIndex":7}],"tool":{"driver":{"fullName":"","informationUri":"https://clang.llvm.org/docs/
// CHECK: UsersManual.html","language":"en-US","name":"clang","rules":[{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"warning","rank":0},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"warning","rank":0},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"{{[0-9]+}}","name":""},{"defaultConfiguration":
// CHECK: {"enabled":true,"level":"error","rank":50},"fullDescription":
// CHECK: {"text":""},"id":"{{[0-9]+}}","name":""}],"version":"{{[0-9]+\.[0-9]+\.[0-9]+[^" ]*}}"}}}],"version":"2.1.0"}
