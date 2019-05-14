// RUN: %clang_cc1 -emit-pch -o %t.1.ast %S/Inputs/struct1.c
// RUN: %clang_cc1 -emit-pch -o %t.2.ast %S/Inputs/struct2.c
// RUN: %clang_cc1 -ast-merge %t.1.ast -ast-merge %t.2.ast -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: struct1.c:13:8: warning: type 'struct S1' has incompatible definitions in different translation units
// CHECK: struct1.c:15:7: note: field 'field2' has type 'int' here
// CHECK: struct2.c:12:9: note: field 'field2' has type 'float' here
// CHECK: struct2.c:15:11: warning: external variable 'x1' declared with incompatible types in different translation units ('struct S1' vs. 'struct S1')
// CHECK: struct1.c:18:11: note: declared here with type 'struct S1'
// CHECK: struct1.c:21:8: warning: type 'struct S2' has incompatible definitions in different translation units
// CHECK: struct2.c:18:7: note: 'S2' is a union here
// CHECK: struct2.c:18:30: warning: external variable 'x2' declared with incompatible types in different translation units ('union S2' vs. 'struct S2')
// CHECK: struct1.c:21:31: note: declared here with type 'struct S2'
// CHECK: struct1.c:24:8: warning: type 'struct S3' has incompatible definitions in different translation units
// CHECK: struct1.c:24:36: note: field 'd' has type 'double' here
// CHECK: struct2.c:21:8: note: no corresponding field here
// CHECK: struct2.c:21:31: warning: external variable 'x3' declared with incompatible types in different translation units ('struct S3' vs. 'struct S3')
// CHECK: struct1.c:24:41: note: declared here with type 'struct S3'
// CHECK: struct1.c:27:8: warning: type 'struct S4' has incompatible definitions in different translation units
// CHECK: struct2.c:24:26: note: field 'f' has type 'float' here
// CHECK: struct1.c:27:8: note: no corresponding field here
// CHECK: struct2.c:24:31: warning: external variable 'x4' declared with incompatible types in different translation units ('struct S4' vs. 'struct S4')
// CHECK: struct1.c:27:22: note: declared here with type 'struct S4'
// CHECK: struct1.c:33:8: warning: type 'struct S6' has incompatible definitions in different translation units
// CHECK: struct1.c:33:33: note: bit-field 'j' with type 'unsigned int' and length 8 here
// CHECK: struct2.c:30:33: note: field 'j' is not a bit-field
// CHECK: struct2.c:30:38: warning: external variable 'x6' declared with incompatible types in different translation units ('struct S6' vs. 'struct S6')
// CHECK: struct1.c:33:42: note: declared here with type 'struct S6'
// CHECK: struct1.c:36:8: warning: type 'struct S7' has incompatible definitions in different translation units
// CHECK: struct1.c:36:33: note: bit-field 'j' with type 'unsigned int' and length 8 here
// CHECK: struct2.c:33:33: note: bit-field 'j' with type 'unsigned int' and length 16 here
// CHECK: struct2.c:33:43: warning: external variable 'x7' declared with incompatible types in different translation units ('struct S7' vs. 'struct S7')
// CHECK: struct1.c:36:42: note: declared here with type 'struct S7'
// CHECK: struct1.c:56:10: warning: type 'struct DeeperError' has incompatible definitions in different translation units
// CHECK: struct1.c:56:35: note: field 'f' has type 'int' here
// CHECK: struct2.c:53:37: note: field 'f' has type 'float' here
// CHECK: struct1.c:54:8: warning: type 'struct DeepError' has incompatible definitions in different translation units
// CHECK: struct1.c:56:41: note: field 'Deeper' has type 'struct DeeperError *' here
// CHECK: struct2.c:53:43: note: field 'Deeper' has type 'struct DeeperError *' here
// CHECK: struct2.c:54:3: warning: external variable 'xDeep' declared with incompatible types in different translation units ('struct DeepError' vs. 'struct DeepError')
// CHECK: struct1.c:57:3: note: declared here with type 'struct DeepError'
// CHECK: struct1.c:74:9: warning: type 'S13' has incompatible definitions in different translation units
// CHECK: struct1.c:75:9: note: field 'i' has type 'Float' (aka 'float') here
// CHECK: struct2.c:72:7: note: field 'i' has type 'int' here
// CHECK: struct2.c:76:5: warning: external variable 'x13' declared with incompatible types in different translation units ('S13' vs. 'S13')
// CHECK: struct1.c:79:5: note: declared here with type 'S13'
// CHECK: struct1.c:130:7: warning: type 'struct DeepUnnamedError::(anonymous at [[PATH_TO_INPUTS:.+]]struct1.c:130:7)' has incompatible definitions in different translation units
// CHECK: struct1.c:131:14: note: field 'i' has type 'long' here
// CHECK: struct2.c:128:15: note: field 'i' has type 'float' here
// CHECK: struct1.c:129:5: warning: type 'union DeepUnnamedError::(anonymous at [[PATH_TO_INPUTS]]struct1.c:129:5)' has incompatible definitions in different translation units
// CHECK: struct1.c:132:9: note: field 'S' has type 'struct (anonymous struct at [[PATH_TO_INPUTS]]struct1.c:130:7)' here
// CHECK: struct2.c:129:9: note: field 'S' has type 'struct (anonymous struct at [[PATH_TO_INPUTS]]struct2.c:127:7)' here
// CHECK: struct2.c:138:3: warning: external variable 'x16' declared with incompatible types in different translation units ('struct DeepUnnamedError' vs. 'struct DeepUnnamedError')
// CHECK: struct1.c:141:3: note: declared here with type 'struct DeepUnnamedError'
// CHECK: 20 warnings generated
