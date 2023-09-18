// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@interface A
@end

@interface A(N)
@end

@protocol M
@end

namespace N { }
namespace M { }
