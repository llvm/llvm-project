#include "header-guard/include/correct.hpp"
#include "header-guard/include/missing.hpp"
#include "header-guard/include/wrong.hpp"

#include "header-guard/other/correct.hpp"
#include "header-guard/other/missing.hpp"
#include "header-guard/other/wrong.hpp"

// ---------------------------------------
// TEST 1: Use no config options (default)
// ---------------------------------------
// RUN: %check_clang_tidy %s llvm-header-guard %t -export-fixes=%t.1.yaml --header-filter=.* -- -I%S > %t.1.msg 2>&1
// RUN: FileCheck -input-file=%t.1.msg -check-prefix=CHECK-MESSAGES1 %s
// RUN: FileCheck -input-file=%t.1.yaml -check-prefix=CHECK-YAML1 %s

// CHECK-MESSAGES1: header-guard/include/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES1: header-guard/other/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES1: header-guard/include/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES1: header-guard/other/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]

// CHECK-YAML1: Message:         header is missing header guard
// CHECK-YAML1: FilePath:        '{{.*}}/header-guard/include/missing.hpp'
// CHECK-YAML1: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML1: ReplacementText: "\n#endif\n"

// CHECK-YAML1: Message:         header guard does not follow preferred style
// CHECK-YAML1: FilePath:        '{{.*}}/header-guard/include/wrong.hpp'
// CHECK-YAML1: ReplacementText: WRONG_HPP

// CHECK-YAML1: Message:         header is missing header guard
// CHECK-YAML1: FilePath:        '{{.*}}/header-guard/other/missing.hpp'
// CHECK-YAML1: ReplacementText: "#ifndef LLVM_HEADER_GUARD_OTHER_MISSING_HPP\n#define LLVM_HEADER_GUARD_OTHER_MISSING_HPP\n\n"
// CHECK-YAML1: ReplacementText: "\n#endif\n"
// CHECK-YAML1: Message:         header guard does not follow preferred style
// CHECK-YAML1: FilePath:        '{{.*}}/header-guard/other/wrong.hpp'
// CHECK-YAML1: ReplacementText: LLVM_HEADER_GUARD_OTHER_WRONG_HPP

// ---------------------------------------
// TEST 2: Set option HeaderDirs=other
// ---------------------------------------
// RUN: %check_clang_tidy %s llvm-header-guard %t -export-fixes=%t.2.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     llvm-header-guard.HeaderDirs: other, \
// RUN:   }}' -- -I%S > %t.2.msg 2>&1
// RUN: FileCheck -input-file=%t.2.msg -check-prefix=CHECK-MESSAGES2 %s
// RUN: FileCheck -input-file=%t.2.yaml -check-prefix=CHECK-YAML2 %s

// CHECK-MESSAGES2: header-guard/include/correct.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/other/correct.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/include/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/other/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/include/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/other/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/include/correct.hpp'
// CHECK-YAML2: ReplacementText: LLVM_HEADER_GUARD_INCLUDE_CORRECT_HPP
// CHECK-YAML2: Message:         header is missing header guard
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/include/missing.hpp'
// CHECK-YAML2: ReplacementText: "#ifndef LLVM_HEADER_GUARD_INCLUDE_MISSING_HPP\n#define LLVM_HEADER_GUARD_INCLUDE_MISSING_HPP\n\n"
// CHECK-YAML2: ReplacementText: "\n#endif\n"
// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/include/wrong.hpp'
// CHECK-YAML2: ReplacementText: LLVM_HEADER_GUARD_INCLUDE_WRONG_HPP

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/other/correct.hpp'
// CHECK-YAML2: ReplacementText: CORRECT_HPP
// CHECK-YAML2: Message:         header is missing header guard
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/other/missing.hpp'
// CHECK-YAML2: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML2: ReplacementText: "\n#endif\n"
// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*}}/header-guard/other/wrong.hpp'
// CHECK-YAML2: ReplacementText: WRONG_HPP


// ---------------------------------------
// TEST 3: Set option HeaderDirs=include;other
// ---------------------------------------
// RUN: %check_clang_tidy %s llvm-header-guard %t -export-fixes=%t.3.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     llvm-header-guard.HeaderDirs: include;other, \
// RUN:   }}' -- -I%S > %t.3.msg 2>&1
// RUN: FileCheck -input-file=%t.3.msg -check-prefix=CHECK-MESSAGES3 %s
// RUN: FileCheck -input-file=%t.3.yaml -check-prefix=CHECK-YAML3 %s

// CHECK-MESSAGES2: header-guard/include/correct.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES2: header-guard/other/correct.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES3: header-guard/include/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES3: header-guard/other/missing.hpp:1:1: warning: header is missing header guard [llvm-header-guard]
// CHECK-MESSAGES3: header-guard/include/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]
// CHECK-MESSAGES3: header-guard/other/wrong.hpp:1:9: warning: header guard does not follow preferred style [llvm-header-guard]

// CHECK-YAML3: Message:         header is missing header guard
// CHECK-YAML3: FilePath:        '{{.*}}/header-guard/include/missing.hpp'
// CHECK-YAML3: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML3: ReplacementText: "\n#endif\n"
// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*}}/header-guard/include/wrong.hpp'
// CHECK-YAML3: ReplacementText: WRONG_HPP

// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*}}/header-guard/other/correct.hpp'
// CHECK-YAML3: ReplacementText: CORRECT_HPP
// CHECK-YAML3: Message:         header is missing header guard
// CHECK-YAML3: FilePath:        '{{.*}}/header-guard/other/missing.hpp'
// CHECK-YAML3: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML3: ReplacementText: "\n#endif\n"
// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*}}/header-guard/other/wrong.hpp'
// CHECK-YAML3: ReplacementText: WRONG_HPP



