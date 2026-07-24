#include "header-guard/include/correct.hpp"
#include "header-guard/include/missing.hpp"
#include "header-guard/include/wrong.hpp"

#include "header-guard/include/other/correct.hpp"
#include "header-guard/include/other/missing.hpp"
#include "header-guard/include/other/wrong.hpp"

// ---------------------------------------
// TEST 1: Use no config options (default)
// ---------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.1.yaml --header-filter=.* -- -I%S > %t.1.msg 2>&1
// RUN: FileCheck -input-file=%t.1.msg -check-prefix=CHECK-MESSAGES1 %s
// RUN: FileCheck -input-file=%t.1.yaml -check-prefix=CHECK-YAML1 %s

// CHECK-MESSAGES1: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES1: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES1: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES1: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML1: Message:         header is missing header guard
// CHECK-YAML1: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML1: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML1: ReplacementText: "\n#endif\n"

// CHECK-YAML1: Message:         header is missing header guard
// CHECK-YAML1: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML1: ReplacementText: "#ifndef OTHER_MISSING_HPP\n#define OTHER_MISSING_HPP\n\n"
// CHECK-YAML1: ReplacementText: "\n#endif\n"

// CHECK-YAML1: Message:         header guard does not follow preferred style
// CHECK-YAML1: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML1: ReplacementText: OTHER_WRONG_HPP

// CHECK-YAML1: Message:         header guard does not follow preferred style
// CHECK-YAML1: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML1: ReplacementText: WRONG_HPP


// ---------------------------------------
// TEST 2: Set option HeaderDirs=other
// ---------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.2.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.HeaderDirs: other, \
// RUN:   }}' -- -I%S > %t.2.msg 2>&1
// RUN: FileCheck -input-file=%t.2.msg -check-prefix=CHECK-MESSAGES2 %s
// RUN: FileCheck -input-file=%t.2.yaml -check-prefix=CHECK-YAML2 %s

// CHECK-MESSAGES2: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES2: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES2: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES2: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES2: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES2: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.}}correct.hpp'
// CHECK-YAML2: ReplacementText: {{.*}}CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_CORRECT_HPP

// CHECK-YAML2: Message:         header is missing header guard
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML2: ReplacementText: "#ifndef {{.*}}CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_MISSING_HPP\n#define {{.*}}CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_MISSING_HPP\n\n"
// CHECK-YAML2: ReplacementText: "\n#endif\n"

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.other.}}correct.hpp'
// CHECK-YAML2: ReplacementText: CORRECT_HPP

// CHECK-YAML2: Message:         header is missing header guard
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML2: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML2: ReplacementText: "\n#endif\n"

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML2: ReplacementText: WRONG_HPP

// CHECK-YAML2: Message:         header guard does not follow preferred style
// CHECK-YAML2: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML2: ReplacementText: {{.*}}CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_WRONG_HPP


// ---------------------------------------
// TEST 3: Set option HeaderDirs=other;include
// ---------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.3.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.HeaderDirs: other;include, \
// RUN:   }}' -- -I%S > %t.3.msg 2>&1
// RUN: FileCheck -input-file=%t.3.msg -check-prefix=CHECK-MESSAGES3 %s
// RUN: FileCheck -input-file=%t.3.yaml -check-prefix=CHECK-YAML3 %s

// CHECK-MESSAGES3: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES3: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES3: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES3: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES3: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES3: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML3: Message:         header is missing header guard
// CHECK-YAML3: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML3: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML3: ReplacementText: "\n#endif\n"

// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*header-guard.include.other.}}correct.hpp'
// CHECK-YAML3: ReplacementText: CORRECT_HPP

// CHECK-YAML3: Message:         header is missing header guard
// CHECK-YAML3: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML3: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML3: ReplacementText: "\n#endif\n"

// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML3: ReplacementText: WRONG_HPP

// CHECK-YAML3: Message:         header guard does not follow preferred style
// CHECK-YAML3: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML3: ReplacementText: WRONG_HPP


// -------------------------------------------------------------------
// TEST 4: Set option HeaderDirs=other;include and Prefix=SOME_PREFIX_
// -------------------------------------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.4.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.Prefix: SOME_PREFIX_, \
// RUN:   }}' -- -I%S > %t.4.msg 2>&1
// RUN: FileCheck -input-file=%t.4.msg -check-prefix=CHECK-MESSAGES4 %s
// RUN: FileCheck -input-file=%t.4.yaml -check-prefix=CHECK-YAML4 %s

// CHECK-MESSAGES4: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES4: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES4: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES4: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES4: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES4: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML4: Message:         header guard does not follow preferred style
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.}}correct.hpp'
// CHECK-YAML4: ReplacementText: SOME_PREFIX_CORRECT_HPP

// CHECK-YAML4: Message:         header is missing header guard
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML4: ReplacementText: "#ifndef SOME_PREFIX_MISSING_HPP\n#define SOME_PREFIX_MISSING_HPP\n\n"
// CHECK-YAML4: ReplacementText: "\n#endif\n"

// CHECK-YAML4: Message:         header guard does not follow preferred style
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.other.}}correct.hpp'
// CHECK-YAML4: ReplacementText: SOME_PREFIX_OTHER_CORRECT_HPP

// CHECK-YAML4: Message:         header is missing header guard
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML4: ReplacementText: "#ifndef SOME_PREFIX_OTHER_MISSING_HPP\n#define SOME_PREFIX_OTHER_MISSING_HPP\n\n"
// CHECK-YAML4: ReplacementText: "\n#endif\n"

// CHECK-YAML4: Message:         header guard does not follow preferred style
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML4: ReplacementText: SOME_PREFIX_OTHER_WRONG_HPP

// CHECK-YAML4: Message:         header guard does not follow preferred style
// CHECK-YAML4: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML4: ReplacementText: SOME_PREFIX_WRONG_HPP



// -------------------------------------------------------------------
// TEST 5: Set option EndifComment=true
// -------------------------------------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.5.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.EndifComment: true, \
// RUN:   }}' -- -I%S > %t.5.msg 2>&1
// RUN: FileCheck -input-file=%t.5.msg -check-prefix=CHECK-MESSAGES5 %s
// RUN: FileCheck -input-file=%t.5.yaml -check-prefix=CHECK-YAML5 %s

// CHECK-MESSAGES5: correct.hpp:6:2: warning: #endif for a header guard should reference the guard macro in a comment [misc-header-guard]
// CHECK-MESSAGES5: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES5: other{{.}}correct.hpp:6:2: warning: #endif for a header guard should reference the guard macro in a comment [misc-header-guard]
// CHECK-MESSAGES5: other{{.}}missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES5: other{{.}}wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES5: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML5: Message:         '#endif for a header guard should reference the guard macro in a comment'
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.}}correct.hpp'
// CHECK-YAML5: ReplacementText: 'endif // CORRECT_HPP'

// CHECK-YAML5: Message:         header is missing header guard
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML5: ReplacementText: "#ifndef MISSING_HPP\n#define MISSING_HPP\n\n"
// CHECK-YAML5: ReplacementText: "\n#endif // MISSING_HPP\n"

// CHECK-YAML5: Message:         '#endif for a header guard should reference the guard macro in a comment'
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.other.}}correct.hpp'
// CHECK-YAML5: ReplacementText: 'endif // OTHER_CORRECT_HPP'

// CHECK-YAML5: Message:         header is missing header guard
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML5: ReplacementText: "#ifndef OTHER_MISSING_HPP\n#define OTHER_MISSING_HPP\n\n"
// CHECK-YAML5: ReplacementText: "\n#endif // OTHER_MISSING_HPP\n"

// CHECK-YAML5: Message:         header guard does not follow preferred style
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML5: ReplacementText: OTHER_WRONG_HPP

// CHECK-YAML5: Message:         header guard does not follow preferred style
// CHECK-YAML5: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML5: ReplacementText: WRONG_HPP

// -------------------------------------------------------------------
// TEST 6: Set option HeaderDirs=path/to/non-matching-dir
// -------------------------------------------------------------------
// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.6.yaml --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.HeaderDirs: path/to/non-matching-dir, \
// RUN:   }}' -- -I%S > %t.6.msg 2>&1
// RUN: FileCheck -input-file=%t.6.msg -check-prefix=CHECK-MESSAGES6 %s
// RUN: FileCheck -input-file=%t.6.yaml -check-prefix=CHECK-YAML6 %s

// CHECK-MESSAGES6: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES6: correct.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES6: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES6: missing.hpp:1:1: warning: header is missing header guard [misc-header-guard]
// CHECK-MESSAGES6: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]
// CHECK-MESSAGES6: wrong.hpp:1:9: warning: header guard does not follow preferred style [misc-header-guard]

// CHECK-YAML6: Message:         header guard does not follow preferred style
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.}}correct.hpp'
// CHECK-YAML6: ReplacementText: {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_CORRECT_HPP

// CHECK-YAML6: Message:         header is missing header guard
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.}}missing.hpp'
// CHECK-YAML6: ReplacementText: "#ifndef {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_MISSING_HPP

// CHECK-YAML6: Message:         header guard does not follow preferred style
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.other.}}correct.hpp'
// CHECK-YAML6: ReplacementText: {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_OTHER_CORRECT_HPP

// CHECK-YAML6: Message:         header is missing header guard
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.other.}}missing.hpp'
// CHECK-YAML6: ReplacementText: "#ifndef {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_OTHER_MISSING_HPP

// CHECK-YAML6: Message:         header guard does not follow preferred style
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.other.}}wrong.hpp'
// CHECK-YAML6: ReplacementText: {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_OTHER_WRONG_HPP

// CHECK-YAML6: Message:         header guard does not follow preferred style
// CHECK-YAML6: FilePath:        '{{.*header-guard.include.}}wrong.hpp'
// CHECK-YAML6: ReplacementText: {{.*}}_CLANG_TOOLS_EXTRA_TEST_CLANG_TIDY_CHECKERS_MISC_HEADER_GUARD_INCLUDE_WRONG_HPP

