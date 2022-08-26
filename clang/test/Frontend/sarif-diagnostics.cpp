// RUN: %clang -fsyntax-only -Wall -Wextra -fdiagnostics-format=sarif %s > %t 2>&1 || true
// RUN: FileCheck -dump-input=always %s --input-file=%t
// CHECK: warning: diagnostic formatting in SARIF mode is currently unstable [-Wsarif-format-unstable]
// CHECK: {"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json","runs":[{"artifacts":[{"length":
// Omit exact length of this file
// CHECK: ,"location":{"index":0,"uri":"file://
// Omit filepath to llvm project directory
// CHECK: clang/test/Frontend/sarif-diagnostics.cpp"},"mimeType":"text/plain","roles":["resultFile"]}],"columnKind":"unicodeCodePoints","results":[{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":1,"startColumn":1,"startLine":12}}}],"message":{"text":"'main' must return 'int'"},"ruleId":"3465","ruleIndex":0},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":11,"startColumn":11,"startLine":13}}}],"message":{"text":"use of undeclared identifier 'hello'"},"ruleId":"4604","ruleIndex":1},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":17,"startColumn":17,"startLine":15}}}],"message":{"text":"invalid digit 'a' in decimal constant"},"ruleId":"898","ruleIndex":2},{"level":"warning","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":5,"startColumn":5,"startLine":19}}}],"message":{"text":"misleading indentation; statement is not part of the previous 'if'"},"ruleId":"1806","ruleIndex":3},{"level":"note","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":3,"startColumn":3,"startLine":17}}}],"message":{"text":"previous statement is here"},"ruleId":"1730","ruleIndex":4},{"level":"warning","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":10,"startColumn":10,"startLine":18}}}],"message":{"text":"unused variable 'Yes'"},"ruleId":"6539","ruleIndex":5},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":12,"startColumn":12,"startLine":21}}}],"message":{"text":"use of undeclared identifier 'hi'"},"ruleId":"4604","ruleIndex":6},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":1,"startColumn":1,"startLine":23}}}],"message":{"text":"extraneous closing brace ('}')"},"ruleId":"1399","ruleIndex":7},{"level":"error","locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":6,"endLine":27,"startColumn":5,"startLine":27}}},{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":10,"endLine":27,"startColumn":9,"startLine":27}}},{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":7,"startColumn":7,"startLine":27}}}],"message":{"text":"invalid operands to binary expression ('t1' and 't1')"},"ruleId":"4539","ruleIndex":8}],"tool":{"driver":{"fullName":"","informationUri":"https://clang.llvm.org/docs/UsersManual.html","language":"en-US","name":"clang","rules":[{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"3465","name":""},{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"4604","name":""},{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"898","name":""},{"defaultConfiguration":{"enabled":true,"level":"warning","rank":-1},"fullDescription":{"text":""},"id":"1806","name":""},{"defaultConfiguration":{"enabled":true,"level":"note","rank":-1},"fullDescription":{"text":""},"id":"1730","name":""},{"defaultConfiguration":{"enabled":true,"level":"warning","rank":-1},"fullDescription":{"text":""},"id":"6539","name":""},{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"4604","name":""},{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"1399","name":""},{"defaultConfiguration":{"enabled":true,"level":"error","rank":50},"fullDescription":{"text":""},"id":"4539","name":""}],"version":"16.0.0"}}}],"version":"2.1.0"}
// CHECK: 2 warnings and 6 errors generated.


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
