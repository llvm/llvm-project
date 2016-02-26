// RUN: %check_clang_tidy %s modernize-deprecated-headers %t -- -extra-arg-before=-isystem%S/Inputs/modernize-deprecated-headers -- -std=c++03 -v

#include <assert.h>
#include <complex.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <iso646.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>
#include <wctype.h>

// Headers deprecated since C++11: expect no diagnostics.
#include <fenv.h>
#include <stdalign.h>
#include <stdbool.h>
#include <tgmath.h>
#include <uchar.h>

// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead [modernize-deprecated-headers]
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'complex.h'; consider using 'ccomplex' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'ctype.h'; consider using 'cctype' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'errno.h'; consider using 'cerrno' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'float.h'; consider using 'cfloat' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'inttypes.h'; consider using 'cinttypes' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'iso646.h'; consider using 'ciso646' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'limits.h'; consider using 'climits' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'locale.h'; consider using 'clocale' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'math.h'; consider using 'cmath' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'setjmp.h'; consider using 'csetjmp' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'signal.h'; consider using 'csignal' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdarg.h'; consider using 'cstdarg' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stddef.h'; consider using 'cstddef' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdint.h'; consider using 'cstdint' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdio.h'; consider using 'cstdio' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdlib.h'; consider using 'cstdlib' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'string.h'; consider using 'cstring' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'time.h'; consider using 'ctime' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'wchar.h'; consider using 'cwchar' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'wctype.h'; consider using 'cwctype' instead

// CHECK-FIXES: #include <cassert>
// CHECK-FIXES: #include <ccomplex>
// CHECK-FIXES: #include <cctype>
// CHECK-FIXES: #include <cerrno>
// CHECK-FIXES: #include <cfloat>
// CHECK-FIXES: #include <cinttypes>
// CHECK-FIXES: #include <ciso646>
// CHECK-FIXES: #include <climits>
// CHECK-FIXES: #include <clocale>
// CHECK-FIXES: #include <cmath>
// CHECK-FIXES: #include <csetjmp>
// CHECK-FIXES: #include <csignal>
// CHECK-FIXES: #include <cstdarg>
// CHECK-FIXES: #include <cstddef>
// CHECK-FIXES: #include <cstdint>
// CHECK-FIXES: #include <cstdio>
// CHECK-FIXES: #include <cstdlib>
// CHECK-FIXES: #include <cstring>
// CHECK-FIXES: #include <ctime>
// CHECK-FIXES: #include <cwchar>
// CHECK-FIXES: #include <cwctype>

#include "assert.h"
#include "complex.h"
#include "ctype.h"
#include "errno.h"
#include "float.h"
#include "inttypes.h"
#include "iso646.h"
#include "limits.h"
#include "locale.h"
#include "math.h"
#include "setjmp.h"
#include "signal.h"
#include "stdarg.h"
#include "stddef.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "wchar.h"
#include "wctype.h"

// Headers deprecated since C++11; expect no diagnostics
#include "fenv.h"
#include "stdalign.h"
#include "stdbool.h"
#include "tgmath.h"
#include "uchar.h"

// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'assert.h'; consider using 'cassert' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'complex.h'; consider using 'ccomplex' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'ctype.h'; consider using 'cctype' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'errno.h'; consider using 'cerrno' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'float.h'; consider using 'cfloat' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'inttypes.h'; consider using 'cinttypes' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'iso646.h'; consider using 'ciso646' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'limits.h'; consider using 'climits' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'locale.h'; consider using 'clocale' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'math.h'; consider using 'cmath' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'setjmp.h'; consider using 'csetjmp' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'signal.h'; consider using 'csignal' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdarg.h'; consider using 'cstdarg' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stddef.h'; consider using 'cstddef' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdint.h'; consider using 'cstdint' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdio.h'; consider using 'cstdio' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'stdlib.h'; consider using 'cstdlib' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'string.h'; consider using 'cstring' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'time.h'; consider using 'ctime' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'wchar.h'; consider using 'cwchar' instead
// CHECK-MESSAGES: :[[@LINE-29]]:10: warning: inclusion of deprecated C++ header 'wctype.h'; consider using 'cwctype' instead

// CHECK-FIXES: #include <cassert>
// CHECK-FIXES: #include <ccomplex>
// CHECK-FIXES: #include <cctype>
// CHECK-FIXES: #include <cerrno>
// CHECK-FIXES: #include <cfloat>
// CHECK-FIXES: #include <cinttypes>
// CHECK-FIXES: #include <ciso646>
// CHECK-FIXES: #include <climits>
// CHECK-FIXES: #include <clocale>
// CHECK-FIXES: #include <cmath>
// CHECK-FIXES: #include <csetjmp>
// CHECK-FIXES: #include <csignal>
// CHECK-FIXES: #include <cstdarg>
// CHECK-FIXES: #include <cstddef>
// CHECK-FIXES: #include <cstdint>
// CHECK-FIXES: #include <cstdio>
// CHECK-FIXES: #include <cstdlib>
// CHECK-FIXES: #include <cstring>
// CHECK-FIXES: #include <ctime>
// CHECK-FIXES: #include <cwchar>
// CHECK-FIXES: #include <cwctype>
